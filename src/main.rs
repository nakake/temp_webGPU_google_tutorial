
use std::{sync::Arc, time::{Duration, SystemTime}};

use wgpu::{util::DeviceExt, BindGroup, Device, Instance, QuerySet, Queue, ShaderStages, Surface, SurfaceConfiguration};
use winit::{dpi::PhysicalSize, event::{ElementState, Event, KeyEvent, WindowEvent}, event_loop::EventLoop, keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowBuilder}};


fn main() {
    pollster::block_on(run());
}

const WINDOW_SIZE: PhysicalSize<i32> = PhysicalSize::new(512, 512);

async fn run() {
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    let _ = window.request_inner_size(WINDOW_SIZE);

    let mut state = State::new(window.clone()).await;

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    event_loop.run(move |event, elwt|
        match event {
            Event::WindowEvent { window_id, event } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested 
                    | WindowEvent::KeyboardInput {
                        event: KeyEvent {
                            physical_key: PhysicalKey::Code(KeyCode::Escape),
                            state: ElementState::Pressed,
                            repeat: false,
                            ..
                        },
                        ..
                    } => {
                        elwt.exit();
                    }
                    WindowEvent::Resized(new_size) => {
                        state.resize(new_size);
                    }
                    WindowEvent::RedrawRequested => {
                        state.compute().unwrap();
                        match state.render() {
                            Ok(_) => {},
                            Err(wgpu::SurfaceError::Lost) => {}
                            Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                            Err(e) => {},
                        }
                    }
                    _ => (),
                }
            } 
            Event::AboutToWait => {
                state.tick();
                window.request_redraw();
            }
            _ => (),
        }
    ).unwrap();
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                }
            ]
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex { position: [-0.8, -0.8, 0.0], color: [1.0, 0.0, 0.0] },
    Vertex { position: [ 0.8, -0.8, 0.0], color: [1.0, 0.0, 0.0] },
    Vertex { position: [ 0.8,  0.8, 0.0], color: [1.0, 0.0, 0.0] },

    Vertex { position: [-0.8, -0.8, 0.0], color: [1.0, 0.0, 0.0] },
    Vertex { position: [ 0.8,  0.8, 0.0], color: [1.0, 0.0, 0.0] },
    Vertex { position: [-0.8,  0.8, 0.0], color: [1.0, 0.0, 0.0] },
];

const GRID_SIZE: f32 = 32.0;

struct State {
    window: Arc<Window>,
    instance: Instance,
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    compute_pipeline: wgpu::ComputePipeline,
    vertex_buffer: wgpu::Buffer,
    bind_groups: [wgpu::BindGroup; 2],
    compute_bind_groups: [wgpu::BindGroup; 2],
    time: SystemTime,
    step: u64,
}

impl State {
    async fn new (window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::default();

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions{
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                    &wgpu::DeviceDescriptor{
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .unwrap();

        let config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();

        surface.configure(&device, &config);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX
        });

        let grid_array: &[f32; 2] = &[GRID_SIZE, GRID_SIZE];

        let grid_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid buffer"),
            contents: bytemuck::cast_slice(grid_array),
            usage: wgpu::BufferUsages::UNIFORM
        });

        let mut cell_state_array_1: Vec<f32> = Vec::new();

        for i in 0..(GRID_SIZE * GRID_SIZE) as i32  {
            if i % 3 == 0 { 
                cell_state_array_1.push(1.0);
            } else {
                cell_state_array_1.push(0.0);
            }
        }

        let cell_state_buffer_1 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cell state buffer"),
            contents: bytemuck::cast_slice(&cell_state_array_1),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let mut cell_state_array_2: Vec<f32> = Vec::new();

        for i in 0..(GRID_SIZE * GRID_SIZE) as i32  {
            if i % 2 == 0 { 
                cell_state_array_2.push(1.0);
            } else {
                cell_state_array_2.push(0.0);
            }
        }

        let cell_state_buffer_2 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cell state buffer"),
            contents: bytemuck::cast_slice(&cell_state_array_2),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        queue.write_buffer(&cell_state_buffer_2, 0, bytemuck::cast_slice(&cell_state_array_2));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, 
                        min_binding_size: None
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None
                },
            ] 
        });

        let bind_groups: [BindGroup; 2] = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &grid_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &cell_state_buffer_1, 
                            offset: 0, 
                            size: None
                        }),
                    },
                ]
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &grid_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &cell_state_buffer_2, 
                            offset: 0, 
                            size: None
                        }),
                    },
                ]
            }),
        ];
        
        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, 
                        min_binding_size: None
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None
                },
            ] 
        });

        let compute_bind_groups: [BindGroup; 2] = [
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compute bind_group"),
                layout: &compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &grid_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &cell_state_buffer_1, 
                            offset: 0, 
                            size: None
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &cell_state_buffer_2, 
                            offset: 0, 
                            size: None
                        }),
                    },
                ]
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("compute bind_group"),
                layout: &compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &grid_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &cell_state_buffer_2, 
                            offset: 0, 
                            size: None
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &cell_state_buffer_1, 
                            offset: 0, 
                            size: None
                        }),
                    },
                ]
            }),
        ];

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("compute.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[
                    Vertex::desc()
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw, 
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&compute_bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "computeMain",
        });

        let time = SystemTime::now();

        let step = 0;

        Self {
            window,
            instance,
            surface,
            device,
            queue,
            config,
            render_pipeline,
            compute_pipeline,
            vertex_buffer,
            bind_groups,
            compute_bind_groups,
            time,
            step,
        }
    }

    fn compute(&mut self) -> Result<(), wgpu::Error> {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compute encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute pass"),
                timestamp_writes: None,
            });

            let index = self.step % 2;
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &self.compute_bind_groups[index as usize], &[]);

            let work_group_count = GRID_SIZE / 8_f32;
            pass.dispatch_workgroups(work_group_count as u32, work_group_count as u32, 0);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture().unwrap();

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor{
                label: Some("render encoder")
            });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.7,
                            b: 0.3,
                            g: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            let index = self.step % 2;
            let index_usize = index as usize; 
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_groups[index_usize], &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..(GRID_SIZE * GRID_SIZE) as u32);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn tick(&mut self) {
        if self.time.elapsed().unwrap() > Duration::from_millis(1000) {
            self.step += 1;
            self.time = SystemTime::now();
        }
    }
}
