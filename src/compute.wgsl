@group(0) @binding(0) var<uniform> grid: vec2<f32>;
@group(0) @binding(1) var<storage> cellSateIn: array<f32>;
@group(0) @binding(2) var<storage, read_write> cellSateOut: array<f32>;

fn cellIndex(cell: vec2u) -> u32 {
    return cell.y * u32(grid.x) + cell.x;
}

@compute @workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
    if (cellSateIn[cellIndex(cell.xy)] == 1) {
        cellSateOut[cellIndex(cell.xy)] = 0.0;
    } else {
        cellSateOut[cellIndex(cell.xy)] = 1.0;
    }
}