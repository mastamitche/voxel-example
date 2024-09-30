use super::{
    gpu_brickmap::GpuVoxelWorld,
    voxel_world::{SetVoxelDataBindGroup, VoxelData},
    VoxelVolume, BRICK_OFFSET,
};
use bevy::{
    core_pipeline::core_3d::Transparent3d,
    ecs::system::{lifetimeless::*, SystemParamItem},
    pbr::{
        MeshPipeline, MeshPipelineKey, RenderMeshInstances, SetMeshBindGroup, SetMeshViewBindGroup,
    },
    prelude::*,
    render::{
        mesh::{GpuBufferInfo, GpuMesh, Indices, MeshVertexBufferLayoutRef},
        render_asset::{RenderAssetUsages, RenderAssets},
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::*,
        renderer::RenderDevice,
        view::ExtractedView,
        Render, RenderApp, RenderSet,
    },
};
use bytemuck::{Pod, Zeroable};

pub struct VoxelRenderPlugin;

impl Plugin for VoxelRenderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CubeHandle>()
            .add_systems(PostUpdate, add_mesh_handles);
        app.sub_app_mut(RenderApp)
            .add_render_command::<Transparent3d, DrawVoxel>()
            .init_resource::<SpecializedMeshPipelines<VoxelPipeline>>()
            .add_systems(
                Render,
                (
                    queue_custom.in_set(RenderSet::QueueMeshes),
                    (prepare_instance_buffers.in_set(RenderSet::PrepareResources),).chain(),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp).init_resource::<VoxelPipeline>();
    }
}

#[derive(Resource, Deref, DerefMut)]
struct CubeHandle(Handle<Mesh>);

impl FromWorld for CubeHandle {
    fn from_world(world: &mut World) -> Self {
        let mut meshes = world.resource_mut::<Assets<Mesh>>();
        CubeHandle(meshes.add(Mesh::from(Cube)))
    }
}

fn add_mesh_handles(
    voxel_volumes: Query<Entity, With<VoxelVolume>>,
    mut commands: Commands,
    cube_handle: Res<CubeHandle>,
) {
    for entity in voxel_volumes.iter() {
        commands.entity(entity).insert(cube_handle.clone());
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct BrickInstance {
    position: Vec3,
    scale: f32,
    brick: u32,
}

#[derive(Component)]
pub struct InstanceBuffer {
    buffer: Buffer,
    length: usize,
}

fn prepare_instance_buffers(
    mut commands: Commands,
    query: Query<(Entity, &VoxelVolume)>,
    render_device: Res<RenderDevice>,
    gpu_voxel_world: Res<GpuVoxelWorld>,
) {
    let mut brick_istance_data = Vec::new();

    // collect nodes
    gpu_voxel_world.recursive_search(&mut |index, pos, depth| {
        // skip non leaf nodes and empty leaf nodes
        if gpu_voxel_world.brickmap[index] <= BRICK_OFFSET {
            return;
        }

        let position = pos.as_vec3() - (1 << gpu_voxel_world.brickmap_depth - 1) as f32;
        let scale = (1 << gpu_voxel_world.brickmap_depth - depth) as f32;
        if depth > gpu_voxel_world.brickmap_depth {
            error!(
                "depth {} > {}. this is probably really bad",
                depth, gpu_voxel_world.brickmap_depth
            );
            return;
        }
        let brick = gpu_voxel_world.brickmap[index] - BRICK_OFFSET;
        brick_istance_data.push(BrickInstance {
            position,
            scale,
            brick,
        });
    });

    // sort nodes
    let (entity, voxel_volume) = query.single();
    if voxel_volume.sort {
        radsort::sort_by_cached_key(&mut brick_istance_data, |brick_instance| {
            let pos = brick_instance.position + brick_instance.scale / 2.0;
            let mut distance = voxel_volume.streaming_pos.distance(pos);
            if voxel_volume.sort_reverse {
                distance = -distance;
            }
            distance
        });
    }

    let length = brick_istance_data.len();
    let brick_instance_data = bytemuck::cast_slice(brick_istance_data.as_slice());
    let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
        label: Some("instance data buffer"),
        contents: brick_instance_data,
        usage: BufferUsages::VERTEX,
    });
    commands
        .entity(entity)
        .insert(InstanceBuffer { buffer, length });
}

fn queue_custom(
    opaque_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Res<VoxelPipeline>,
    msaa: Res<Msaa>,
    mut pipelines: ResMut<SpecializedMeshPipelines<VoxelPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<GpuMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    voxel_volumes: Query<Entity, With<VoxelVolume>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    mut views: Query<(Entity, &ExtractedView)>,
) {
    let draw_custom = opaque_3d_draw_functions.read().id::<DrawVoxel>();

    let msaa_key = MeshPipelineKey::from_msaa_samples(msaa.samples());
    for (e, view) in &mut views {
        let Some(transparent_phase) = transparent_render_phases.get_mut(&e) else {
            continue;
        };
        let view_key = msaa_key | MeshPipelineKey::from_hdr(view.hdr);
        let rangefinder = view.rangefinder3d();
        for entity in &voxel_volumes {
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(entity) else {
                continue;
            };
            let Some(mesh) = meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };
            let key =
                view_key | MeshPipelineKey::from_primitive_topology(mesh.primitive_topology());
            let pipeline = pipelines
                .specialize(&pipeline_cache, &custom_pipeline, key, &mesh.layout)
                .unwrap();

            let t3d = Transparent3d {
                entity,
                pipeline,
                draw_function: draw_custom,
                distance: rangefinder.distance_translation(&mesh_instance.translation),
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex(0),
            };
            transparent_phase.add(t3d);
            // let value = Opaque3d::new(
            //     key,
            //     entity,
            //     // distance: rangefinder
            //     //     .distance_translation(&mesh_instance.transforms.transform.translation),
            //     0..1,
            //     PhaseItemExtraIndex::dynamic_offset(0),
            // );
            //opaque_phase.add(key, entity, BinnedRenderPhaseType::BatchableMesh);
        }
    }
}

#[derive(Resource)]
pub struct VoxelPipeline {
    shader: Handle<Shader>,
    mesh_pipeline: MeshPipeline,
    voxel_data_bind_group_layout: BindGroupLayout,
}

impl FromWorld for VoxelPipeline {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        let mesh_pipeline = world.resource::<MeshPipeline>().clone();
        let voxel_data = world.resource::<VoxelData>();

        let shader = asset_server.load("instancing.wgsl");
        let voxel_data_bind_group_layout = voxel_data.bind_group_layout.clone();

        VoxelPipeline {
            shader,
            mesh_pipeline,
            voxel_data_bind_group_layout,
        }
    }
}

impl SpecializedMeshPipeline for VoxelPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key, layout)?;

        descriptor.layout = vec![
            self.mesh_pipeline.get_view_layout(key.into()).clone(),
            self.mesh_pipeline.mesh_layouts.model_only.clone(),
            self.voxel_data_bind_group_layout.clone(),
        ];

        // meshes typically live in bind group 2. because we are using bindgroup 1
        // we need to add MESH_BINDGROUP_1 shader def so that the bindings are correctly
        // linked in the shader
        descriptor
            .vertex
            .shader_defs
            .push("MESH_BINDGROUP_1".into());
        descriptor
            .fragment
            .as_mut()
            .unwrap()
            .shader_defs
            .push("MESH_BINDGROUP_1".into());

        descriptor.vertex.shader = self.shader.clone();
        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: std::mem::size_of::<BrickInstance>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 3, // shader locations 0-2 are taken up by Position, Normal and UV attributes
                },
                VertexAttribute {
                    format: VertexFormat::Uint32,
                    offset: 16,
                    shader_location: 4,
                },
            ],
        });

        descriptor.fragment.as_mut().unwrap().shader = self.shader.clone();

        descriptor.primitive.cull_mode = None;

        Ok(descriptor)
    }
}

type DrawVoxel = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshBindGroup<1>,
    SetVoxelDataBindGroup<2>,
    DrawVoxelPhase,
);

pub struct DrawVoxelPhase;

impl<P: PhaseItem> RenderCommand<P> for DrawVoxelPhase {
    type Param = (SRes<RenderAssets<GpuMesh>>, SRes<RenderMeshInstances>);
    type ViewQuery = ();
    type ItemQuery = Read<InstanceBuffer>;

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        instance_buffer: Option<&'w InstanceBuffer>,
        (meshes, render_mesh_instances): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(item.entity())
        else {
            return RenderCommandResult::Failure;
        };
        let Some(gpu_mesh) = meshes.into_inner().get(mesh_instance.mesh_asset_id) else {
            return RenderCommandResult::Failure;
        };

        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        if let Some(instance_buffer) = instance_buffer {
            pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));

            match &gpu_mesh.buffer_info {
                GpuBufferInfo::Indexed {
                    buffer,
                    index_format,
                    count,
                } => {
                    pass.set_index_buffer(buffer.slice(..), 0, *index_format);
                    pass.draw_indexed(0..*count, 0, 0..instance_buffer.length as u32);
                }
                GpuBufferInfo::NonIndexed => {
                    pass.draw(0..gpu_mesh.vertex_count, 0..instance_buffer.length as u32);
                }
            }
        }
        RenderCommandResult::Success
    }
}

// copied from bevy as cuboid only allows half_size not min/max
pub struct Cube;

impl MeshBuilder for Cube {
    fn build(&self) -> Mesh {
        let min = Vec3::ZERO;
        let max = Vec3::ONE;

        // Suppose Y-up right hand, and camera look from +Z to -Z
        let vertices = &[
            // Front
            ([min.x, min.y, max.z], [0.0, 0.0, 1.0], [0.0, 0.0]),
            ([max.x, min.y, max.z], [0.0, 0.0, 1.0], [1.0, 0.0]),
            ([max.x, max.y, max.z], [0.0, 0.0, 1.0], [1.0, 1.0]),
            ([min.x, max.y, max.z], [0.0, 0.0, 1.0], [0.0, 1.0]),
            // Back
            ([min.x, max.y, min.z], [0.0, 0.0, -1.0], [1.0, 0.0]),
            ([max.x, max.y, min.z], [0.0, 0.0, -1.0], [0.0, 0.0]),
            ([max.x, min.y, min.z], [0.0, 0.0, -1.0], [0.0, 1.0]),
            ([min.x, min.y, min.z], [0.0, 0.0, -1.0], [1.0, 1.0]),
            // Right
            ([max.x, min.y, min.z], [1.0, 0.0, 0.0], [0.0, 0.0]),
            ([max.x, max.y, min.z], [1.0, 0.0, 0.0], [1.0, 0.0]),
            ([max.x, max.y, max.z], [1.0, 0.0, 0.0], [1.0, 1.0]),
            ([max.x, min.y, max.z], [1.0, 0.0, 0.0], [0.0, 1.0]),
            // Left
            ([min.x, min.y, max.z], [-1.0, 0.0, 0.0], [1.0, 0.0]),
            ([min.x, max.y, max.z], [-1.0, 0.0, 0.0], [0.0, 0.0]),
            ([min.x, max.y, min.z], [-1.0, 0.0, 0.0], [0.0, 1.0]),
            ([min.x, min.y, min.z], [-1.0, 0.0, 0.0], [1.0, 1.0]),
            // Top
            ([max.x, max.y, min.z], [0.0, 1.0, 0.0], [1.0, 0.0]),
            ([min.x, max.y, min.z], [0.0, 1.0, 0.0], [0.0, 0.0]),
            ([min.x, max.y, max.z], [0.0, 1.0, 0.0], [0.0, 1.0]),
            ([max.x, max.y, max.z], [0.0, 1.0, 0.0], [1.0, 1.0]),
            // Bottom
            ([max.x, min.y, max.z], [0.0, -1.0, 0.0], [0.0, 0.0]),
            ([min.x, min.y, max.z], [0.0, -1.0, 0.0], [1.0, 0.0]),
            ([min.x, min.y, min.z], [0.0, -1.0, 0.0], [1.0, 1.0]),
            ([max.x, min.y, min.z], [0.0, -1.0, 0.0], [0.0, 1.0]),
        ];

        let positions: Vec<_> = vertices.iter().map(|(p, _, _)| *p).collect();
        let normals: Vec<_> = vertices.iter().map(|(_, n, _)| *n).collect();
        let uvs: Vec<_> = vertices.iter().map(|(_, _, uv)| *uv).collect();

        let indices = Indices::U32(vec![
            0, 1, 2, 2, 3, 0, // front
            4, 5, 6, 6, 7, 4, // back
            8, 9, 10, 10, 11, 8, // right
            12, 13, 14, 14, 15, 12, // left
            16, 17, 18, 18, 19, 16, // top
            20, 21, 22, 22, 23, 20, // bottom
        ]);

        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_indices(indices)
    }
}
