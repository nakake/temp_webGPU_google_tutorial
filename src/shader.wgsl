struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) cell: vec2<f32>,
}

@group(0) @binding(0) var<uniform> grid: vec2<f32>;
@group(0) @binding(1) var<storage> cell_state: array<f32>;

@vertex
fn vs_main(
    model: VertexInput,
    @builtin(instance_index) instance: u32
) -> VertexOutput {
    let i = f32(instance);
    let cell = vec2<f32>(i % grid.x, floor(i / grid.x));
    let state = cell_state[instance];

    let cell_offset = cell / grid * 2;
    let grid_pos = ((model.position.xy * state) + 1) / grid - 1 + cell_offset;

    var out: VertexOutput;
    out.color = model.color;
    out.clip_position = vec4<f32>(grid_pos, model.position.z, 1.0);
    out.cell = cell;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let c = in.cell / grid;
    return vec4<f32>(c , 1.0 - c.x, 1.0);
}