const std = @import("std");
const vapoursynth = @import("vapoursynth");

const vs = vapoursynth.vapoursynth4;
const ZAPI = vapoursynth.ZAPI;

const util = @import("../util.zig");

const allocator = std.heap.c_allocator;
const analysis_data_size = 21 * @sizeOf(u32);
const whole_number_tolerance = 0.001;

// These fields are the spatial values ScaleVect needs from MVTools_MVAnalysisData.
const AnalysisData = struct {
    block_size_x: u32,
    block_size_y: u32,
    pel: u32,
    level_count: u32,
    width: u32,
    height: u32,
    overlap_x: u32,
    overlap_y: u32,
    block_count_x: u32,
    block_count_y: u32,
    chroma_ratio_y: u32,
    chroma_ratio_x: u32,
    padding_x: u32,
    padding_y: u32,
};

const AxisGeometry = struct {
    size: u32,
    block_size: u32,
    overlap: u32,
    padding: u32,
    used_fallback: bool,
    finest_count_mismatch: bool,
    pyramid_count_mismatch: bool,
};

const Geometry = struct {
    x: AxisGeometry,
    y: AxisGeometry,
};

const CompatibilityIssues = struct {
    analyse_block_pair: bool = false,
    luma_kernel_pair: bool = false,
    chroma_kernel_pair: bool = false,
    chroma_fractional: bool = false,
    chroma_alignment: bool = false,
    overlap_odd: bool = false,
    overlap_too_large: bool = false,
    finest_count_mismatch: bool = false,
    pyramid_count_mismatch: bool = false,
    coherent_snap_fallback: bool = false,
};

const FunctionData = struct {
    node: ?*vs.Node,
    vi: *const vs.VideoInfo,
    scale_x: f64,
    scale_y: f64,
    geometry: Geometry,
    sad_scale: f64,
};

fn readAnalysisData(data: []const u8) ?AnalysisData {
    if (data.len != analysis_data_size) return null;
    return .{
        .block_size_x = std.mem.readInt(u32, data[2 * @sizeOf(u32) ..][0..4], .little),
        .block_size_y = std.mem.readInt(u32, data[3 * @sizeOf(u32) ..][0..4], .little),
        .pel = std.mem.readInt(u32, data[4 * @sizeOf(u32) ..][0..4], .little),
        .level_count = std.mem.readInt(u32, data[5 * @sizeOf(u32) ..][0..4], .little),
        .width = std.mem.readInt(u32, data[10 * @sizeOf(u32) ..][0..4], .little),
        .height = std.mem.readInt(u32, data[11 * @sizeOf(u32) ..][0..4], .little),
        .overlap_x = std.mem.readInt(u32, data[12 * @sizeOf(u32) ..][0..4], .little),
        .overlap_y = std.mem.readInt(u32, data[13 * @sizeOf(u32) ..][0..4], .little),
        .block_count_x = std.mem.readInt(u32, data[14 * @sizeOf(u32) ..][0..4], .little),
        .block_count_y = std.mem.readInt(u32, data[15 * @sizeOf(u32) ..][0..4], .little),
        .chroma_ratio_y = std.mem.readInt(u32, data[17 * @sizeOf(u32) ..][0..4], .little),
        .chroma_ratio_x = std.mem.readInt(u32, data[18 * @sizeOf(u32) ..][0..4], .little),
        .padding_x = std.mem.readInt(u32, data[19 * @sizeOf(u32) ..][0..4], .little),
        .padding_y = std.mem.readInt(u32, data[20 * @sizeOf(u32) ..][0..4], .little),
    };
}

fn normalizeScale(scale: f64) f64 {
    // Treat factors very close to integers as exact integers to avoid avoidable grid drift.
    const nearest = @round(scale);
    return if (@abs(scale - nearest) <= whole_number_tolerance) nearest else scale;
}

fn roundPositiveU32(value: f64) ?u32 {
    if (!std.math.isFinite(value) or value <= 0 or value > @as(f64, @floatFromInt(std.math.maxInt(i32)))) return null;
    const rounded = @round(value);
    return if (rounded >= 1) @intFromFloat(rounded) else null;
}

fn roundNonNegativeU32(value: f64) ?u32 {
    if (!std.math.isFinite(value) or value < 0 or value > @as(f64, @floatFromInt(std.math.maxInt(i32)))) return null;
    return @intFromFloat(@round(value));
}

fn pyramidBlockCount(block_size: u32, overlap: u32, block_count: u32, level: u32) ?u32 {
    // MVTools reconstructs each coarse level from the finest-level covered extent.
    if (block_size <= overlap or block_count == 0 or level >= 31) return null;
    const step = block_size - overlap;
    const covered = @as(u64, step) * block_count + overlap;
    const level_width = covered >> @intCast(level);
    if (level_width <= overlap) return 0;
    return @intCast((level_width - overlap) / step);
}

fn preservesPyramidCounts(original_block_size: u32, original_overlap: u32, candidate_block_size: u32, candidate_overlap: u32, block_count: u32, level_count: u32) bool {
    if (candidate_block_size <= candidate_overlap) return false;
    var level: u32 = 1;
    while (level < level_count) : (level += 1) {
        const original_count = pyramidBlockCount(original_block_size, original_overlap, block_count, level) orelse return false;
        const candidate_count = pyramidBlockCount(candidate_block_size, candidate_overlap, block_count, level) orelse return false;
        if (original_count != candidate_count) return false;
    }
    return true;
}

fn deriveBlockCount(size: u32, block_size: u32, overlap: u32) ?u32 {
    if (block_size <= overlap or size < overlap) return null;
    return (size - overlap) / (block_size - overlap);
}

fn makeAxisGeometry(size: u32, block_size: u32, overlap: u32, padding: u32, block_count: u32, level_count: u32, scale: f64) ?AxisGeometry {
    const scaled_size = roundPositiveU32(@as(f64, @floatFromInt(size)) * scale) orelse return null;
    const scaled_padding = roundNonNegativeU32(@as(f64, @floatFromInt(padding)) * scale) orelse return null;
    const ideal_block_size = @as(f64, @floatFromInt(block_size)) * scale;
    const ideal_overlap = @as(f64, @floatFromInt(overlap)) * scale;
    const ideal_step = @as(f64, @floatFromInt(block_size - overlap)) * scale;
    _ = roundPositiveU32(ideal_block_size) orelse return null;
    _ = roundNonNegativeU32(ideal_overlap) orelse return null;
    const block_start = @max(@as(i64, 1), @as(i64, @intFromFloat(@floor(ideal_block_size))) - 1);
    const block_end = @as(i64, @intFromFloat(@ceil(ideal_block_size))) + 1;
    const overlap_start = @max(@as(i64, 0), @as(i64, @intFromFloat(@floor(ideal_overlap))) - 1);
    const overlap_end = @as(i64, @intFromFloat(@ceil(ideal_overlap))) + 1;
    var found = false;
    var best_block_size: u32 = 0;
    var best_overlap: u32 = 0;
    var best_cost = std.math.inf(f64);
    var best_step_cost = std.math.inf(f64);
    var best_finest_match = false;
    var candidate_block = block_start;
    // Search a small neighborhood for the closest grid that MVTools will parse with unchanged level counts.
    while (candidate_block <= block_end) : (candidate_block += 1) {
        var candidate_overlap = overlap_start;
        while (candidate_overlap <= overlap_end) : (candidate_overlap += 1) {
            if (candidate_block <= candidate_overlap or candidate_block > std.math.maxInt(i32)) continue;
            const candidate_block_u32: u32 = @intCast(candidate_block);
            const candidate_overlap_u32: u32 = @intCast(candidate_overlap);
            if (!preservesPyramidCounts(block_size, overlap, candidate_block_u32, candidate_overlap_u32, block_count, level_count)) continue;
            const candidate_count = deriveBlockCount(scaled_size, candidate_block_u32, candidate_overlap_u32);
            const candidate_finest_match = candidate_count != null and candidate_count.? == block_count;
            const cost = @abs(@as(f64, @floatFromInt(candidate_block_u32)) - ideal_block_size) + @abs(@as(f64, @floatFromInt(candidate_overlap_u32)) - ideal_overlap);
            const step_cost = @abs(@as(f64, @floatFromInt(candidate_block_u32 - candidate_overlap_u32)) - ideal_step);
            if (!found or (candidate_finest_match and !best_finest_match) or (candidate_finest_match == best_finest_match and (cost < best_cost or (cost == best_cost and step_cost < best_step_cost)))) {
                found = true;
                best_block_size = candidate_block_u32;
                best_overlap = candidate_overlap_u32;
                best_cost = cost;
                best_step_cost = step_cost;
                best_finest_match = candidate_finest_match;
            }
        }
    }
    var used_fallback = false;
    if (!found) {
        best_block_size = roundPositiveU32(ideal_block_size) orelse return null;
        best_overlap = roundNonNegativeU32(ideal_overlap) orelse return null;
        if (best_block_size <= best_overlap) return null;
        used_fallback = true;
    }
    const derived_count = deriveBlockCount(scaled_size, best_block_size, best_overlap);
    const pyramid_mismatch = !preservesPyramidCounts(block_size, overlap, best_block_size, best_overlap, block_count, level_count);
    return .{
        .size = scaled_size,
        .block_size = best_block_size,
        .overlap = best_overlap,
        .padding = scaled_padding,
        .used_fallback = used_fallback,
        .finest_count_mismatch = derived_count == null or derived_count.? != block_count,
        .pyramid_count_mismatch = pyramid_mismatch,
    };
}

fn writeAnalysisData(in: []const u8, out: []u8, geometry: Geometry) void {
    // Preserve non-spatial analysis fields exactly as received.
    @memcpy(out, in);
    std.mem.writeInt(u32, out[2 * @sizeOf(u32) ..][0..4], geometry.x.block_size, .little);
    std.mem.writeInt(u32, out[3 * @sizeOf(u32) ..][0..4], geometry.y.block_size, .little);
    std.mem.writeInt(u32, out[10 * @sizeOf(u32) ..][0..4], geometry.x.size, .little);
    std.mem.writeInt(u32, out[11 * @sizeOf(u32) ..][0..4], geometry.y.size, .little);
    std.mem.writeInt(u32, out[12 * @sizeOf(u32) ..][0..4], geometry.x.overlap, .little);
    std.mem.writeInt(u32, out[13 * @sizeOf(u32) ..][0..4], geometry.y.overlap, .little);
    std.mem.writeInt(u32, out[19 * @sizeOf(u32) ..][0..4], geometry.x.padding, .little);
    std.mem.writeInt(u32, out[20 * @sizeOf(u32) ..][0..4], geometry.y.padding, .little);
}

test {
    const input = [_]u8{ 1, 0, 0, 0 } ** 21;
    var output = [_]u8{ 0, 0, 0, 0 } ** 21;
    const x = AxisGeometry{ .size = 2, .block_size = 2, .overlap = 2, .padding = 2, .used_fallback = false, .finest_count_mismatch = false, .pyramid_count_mismatch = false };
    const y = AxisGeometry{ .size = 4, .block_size = 4, .overlap = 4, .padding = 4, .used_fallback = false, .finest_count_mismatch = false, .pyramid_count_mismatch = false };
    writeAnalysisData(input[0..], output[0..], .{ .x = x, .y = y });
    try std.testing.expectEqual(1, std.mem.readInt(u32, output[0..4], .little));
    try std.testing.expectEqual(2, std.mem.readInt(u32, output[2 * @sizeOf(u32) ..][0..4], .little));
    try std.testing.expectEqual(4, std.mem.readInt(u32, output[3 * @sizeOf(u32) ..][0..4], .little));
}

fn roundScaledInt(comptime T: type, value: T, scale: f64) ?T {
    const scaled = @as(f64, @floatFromInt(value)) * scale;
    if (!std.math.isFinite(scaled)) return null;
    const rounded = @round(scaled);
    if (T == u64) {
        if (rounded < 0 or rounded >= 18446744073709551616.0) return null;
    } else if (rounded < @as(f64, @floatFromInt(std.math.minInt(T))) or rounded > @as(f64, @floatFromInt(std.math.maxInt(T)))) {
        return null;
    }
    return @intFromFloat(rounded);
}

fn scaleVectorData(in: []const u8, out: []u8, scale_x: f64, scale_y: f64, sad_scale: f64) bool {
    if (in.len < 2 * @sizeOf(u32) or in.len > std.math.maxInt(u32)) return false;
    // Copy first so invalid-vector payloads and all size fields remain byte-identical.
    @memcpy(out, in);
    var position: u32 = 0;
    const size, const position_after_size = util.readAndCopyInt(u32, in, out, position);
    position = position_after_size;
    if (in.len != size) return false;
    const validity_int, const position_after_validity = util.readAndCopyInt(u32, in, out, position);
    position = position_after_validity;
    if (validity_int != 1) return true;
    while (position < size) {
        if (size - position < @sizeOf(u32)) return false;
        const level_header_position = position;
        const level_size, const start_position = util.readAndCopyInt(u32, in, out, position);
        if (level_size < @sizeOf(u32) or level_size > size - level_header_position) return false;
        const end_position = level_header_position + level_size;
        position = start_position;
        while (position < end_position) {
            if (end_position - position < 2 * @sizeOf(i32) + @sizeOf(u64)) return false;
            const vector_x, const after_x = util.readInt(i32, in, position);
            const vector_y, const after_y = util.readInt(i32, in, after_x);
            const sad, const after_sad = util.readInt(u64, in, after_y);
            const scaled_x = roundScaledInt(i32, vector_x, scale_x) orelse return false;
            const scaled_y = roundScaledInt(i32, vector_y, scale_y) orelse return false;
            const scaled_sad = roundScaledInt(u64, sad, sad_scale) orelse return false;
            std.mem.writeInt(i32, out[position..][0..4], scaled_x, .little);
            std.mem.writeInt(i32, out[after_x..][0..4], scaled_y, .little);
            std.mem.writeInt(u64, out[after_y..][0..8], scaled_sad, .little);
            position = after_sad;
        }
    }
    return position == size;
}

test {
    const single_vector = [_]u8{ 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0 }; // (1, 2, 3)
    const vectors = single_vector ** 10;
    const level_size = [_]u8{ vectors.len + @sizeOf(u32), 0, 0, 0 };
    const valid = [_]u8{ 1, 0, 0, 0 };
    const total_size = [_]u8{ vectors.len + valid.len + level_size.len + @sizeOf(u32), 0, 0, 0 };
    const input = total_size ++ valid ++ level_size ++ vectors;
    var output = [_]u8{0} ** input.len;
    try std.testing.expect(scaleVectorData(input[0..], output[0..], 2.0, 4.0, 8.0));
    try std.testing.expectEqual(2, std.mem.readInt(i32, output[3 * @sizeOf(u32) ..][0..4], .little));
    try std.testing.expectEqual(8, std.mem.readInt(i32, output[4 * @sizeOf(u32) ..][0..4], .little));
    try std.testing.expectEqual(24, std.mem.readInt(u64, output[5 * @sizeOf(u32) ..][0..8], .little));
}

fn isAnalyseBlockPair(width: u32, height: u32) bool {
    return switch (width) {
        4 => height == 4,
        8 => height == 4 or height == 8,
        16 => height == 2 or height == 8 or height == 16,
        32 => height == 16 or height == 32,
        64 => height == 32 or height == 64,
        128 => height == 64 or height == 128,
        else => false,
    };
}

fn isKernelBlockPair(width: u32, height: u32) bool {
    return switch (width) {
        2 => height == 2 or height == 4,
        4 => height == 2 or height == 4 or height == 8,
        8 => height == 1 or height == 2 or height == 4 or height == 8 or height == 16,
        16 => height == 1 or height == 2 or height == 4 or height == 8 or height == 16 or height == 32,
        32 => height == 8 or height == 16 or height == 32 or height == 64,
        64 => height == 16 or height == 32 or height == 64 or height == 128,
        128 => height == 32 or height == 64 or height == 128,
        else => false,
    };
}

fn collectCompatibilityIssues(analysis: AnalysisData, geometry: Geometry) CompatibilityIssues {
    // Analyse supports fewer luma pairs than the processing kernels support.
    var issues = CompatibilityIssues{};
    issues.analyse_block_pair = !isAnalyseBlockPair(geometry.x.block_size, geometry.y.block_size);
    issues.luma_kernel_pair = !isKernelBlockPair(geometry.x.block_size, geometry.y.block_size);
    const ratios_valid = analysis.chroma_ratio_x > 0 and analysis.chroma_ratio_y > 0;
    issues.chroma_fractional = !ratios_valid or geometry.x.block_size % analysis.chroma_ratio_x != 0 or geometry.y.block_size % analysis.chroma_ratio_y != 0;
    if (!issues.chroma_fractional) {
        issues.chroma_kernel_pair = !isKernelBlockPair(geometry.x.block_size / analysis.chroma_ratio_x, geometry.y.block_size / analysis.chroma_ratio_y);
    }
    if (ratios_valid) {
        const x_alignment_invalid = geometry.x.size % analysis.chroma_ratio_x != 0 or geometry.x.padding % analysis.chroma_ratio_x != 0 or geometry.x.overlap % analysis.chroma_ratio_x != 0;
        const y_alignment_invalid = geometry.y.size % analysis.chroma_ratio_y != 0 or geometry.y.padding % analysis.chroma_ratio_y != 0 or geometry.y.overlap % analysis.chroma_ratio_y != 0;
        issues.chroma_alignment = x_alignment_invalid or y_alignment_invalid;
    }
    issues.overlap_odd = geometry.x.overlap % 2 != 0 or geometry.y.overlap % 2 != 0;
    issues.overlap_too_large = geometry.x.overlap > geometry.x.block_size / 2 or geometry.y.overlap > geometry.y.block_size / 2;
    issues.finest_count_mismatch = geometry.x.finest_count_mismatch or geometry.y.finest_count_mismatch;
    issues.pyramid_count_mismatch = geometry.x.pyramid_count_mismatch or geometry.y.pyramid_count_mismatch;
    issues.coherent_snap_fallback = geometry.x.used_fallback or geometry.y.used_fallback;
    return issues;
}

fn appendWarning(buffer: []u8, position: *usize, text: []const u8) void {
    const separator = if (position.* == 0) "ScaleVect: output may not be compatible with MVTools: " else "; ";
    if (position.* + separator.len + text.len + 1 > buffer.len) return;
    @memcpy(buffer[position.*..][0..separator.len], separator);
    position.* += separator.len;
    @memcpy(buffer[position.*..][0..text.len], text);
    position.* += text.len;
}

fn logCompatibilityWarning(issues: CompatibilityIssues, core: ?*vs.Core, vsapi: ?*const vs.API) void {
    // Report every detected compatibility concern in one creation-time warning.
    var buffer: [2048]u8 = undefined;
    var position: usize = 0;
    if (issues.analyse_block_pair) appendWarning(&buffer, &position, "the luma block pair cannot be generated by Analyse");
    if (issues.luma_kernel_pair) appendWarning(&buffer, &position, "the luma block pair has no Compensate/Degrain kernel");
    if (issues.chroma_fractional) appendWarning(&buffer, &position, "the block dimensions do not divide cleanly into the stored chroma subsampling");
    if (issues.chroma_kernel_pair) appendWarning(&buffer, &position, "the chroma block pair has no Compensate/Degrain kernel");
    if (issues.chroma_alignment) appendWarning(&buffer, &position, "scaled dimensions, padding, or overlap violate chroma-subsampling alignment");
    if (issues.overlap_odd) appendWarning(&buffer, &position, "scaled overlap is odd");
    if (issues.overlap_too_large) appendWarning(&buffer, &position, "scaled overlap exceeds half of the block size");
    if (issues.finest_count_mismatch) appendWarning(&buffer, &position, "scaled frame geometry does not reproduce the stored finest-level block count");
    if (issues.pyramid_count_mismatch) appendWarning(&buffer, &position, "scaled block geometry changes one or more pyramid-level block counts");
    if (issues.coherent_snap_fallback) appendWarning(&buffer, &position, "no nearby coherent block grid was found and nearest-value rounding was used");
    if (position == 0) return;
    buffer[position] = 0;
    vsapi.?.logMessage.?(.Warning, buffer[0..position :0], core);
}

export fn getFrameScaleVect(n: c_int, activation_reason: vs.ActivationReason, instance_data: ?*anyopaque, frame_data: ?*?*anyopaque, frame_ctx: ?*vs.FrameContext, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) ?*const vs.Frame {
    _ = frame_data;
    const d: *FunctionData = @ptrCast(@alignCast(instance_data));
    const zapi = ZAPI.init(vsapi, core, frame_ctx);
    if (activation_reason == .Initial) {
        vsapi.?.requestFrameFilter.?(n, d.node, frame_ctx);
    } else if (activation_reason == .AllFramesReady) {
        var src = zapi.initZFrame(d.node, n);
        var dst = src.copyFrame();
        defer src.deinit();
        const src_props = src.getPropertiesRO();
        const dst_props = dst.getPropertiesRW();
        const analysis_data_in = src_props.getData("MVTools_MVAnalysisData", 0) orelse {
            vsapi.?.setFilterError.?("Could not read MVTools_MVAnalysisData property when attempting to scale vectors.", frame_ctx);
            dst.deinit();
            return null;
        };
        if (analysis_data_in.len != analysis_data_size) {
            vsapi.?.setFilterError.?("MVTools_MVAnalysisData has an unexpected size.", frame_ctx);
            dst.deinit();
            return null;
        }
        const analysis_data_out = allocator.allocSentinel(u8, analysis_data_in.len, 0) catch {
            vsapi.?.setFilterError.?("Out of memory", frame_ctx);
            dst.deinit();
            return null;
        };
        defer allocator.free(analysis_data_out);
        writeAnalysisData(analysis_data_in, analysis_data_out[0..analysis_data_in.len], d.geometry);
        dst_props.setData("MVTools_MVAnalysisData", analysis_data_out, .Binary, .Replace);
        const vector_data_in = src_props.getData("MVTools_vectors", 0) orelse {
            vsapi.?.setFilterError.?("Could not read MVTools_vectors property when attempting to scale vectors.", frame_ctx);
            dst.deinit();
            return null;
        };
        const vector_data_out = allocator.alloc(u8, vector_data_in.len + 1) catch {
            vsapi.?.setFilterError.?("Out of memory", frame_ctx);
            dst.deinit();
            return null;
        };
        defer allocator.free(vector_data_out);
        if (!scaleVectorData(vector_data_in, vector_data_out[0..vector_data_in.len], d.scale_x, d.scale_y, d.sad_scale)) {
            vsapi.?.setFilterError.?("Could not safely scale MVTools_vectors because its data is malformed or a scaled value overflowed.", frame_ctx);
            dst.deinit();
            return null;
        }
        vector_data_out[vector_data_in.len] = 0;
        dst_props.setData("MVTools_vectors", vector_data_out[0..vector_data_in.len :0], .Binary, .Replace);
        return dst.frame;
    }
    return null;
}

export fn freeScaleVect(instance_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    _ = core;
    const d: *FunctionData = @ptrCast(@alignCast(instance_data));
    vsapi.?.freeNode.?(d.node);
    allocator.destroy(d);
}

pub export fn createScaleVect(in: ?*const vs.Map, out: ?*vs.Map, user_data: ?*anyopaque, core: ?*vs.Core, vsapi: ?*const vs.API) callconv(.c) void {
    _ = user_data;
    var d: FunctionData = undefined;
    const zapi = ZAPI.init(vsapi, core, null);
    var map_in = zapi.initZMap(in);
    var map_out = zapi.initZMap(out);
    d.node, d.vi = map_in.getNodeVi("clip").?;
    d.scale_x = normalizeScale(map_in.getFloat(f64, "scaleX") orelse 1.0);
    d.scale_y = normalizeScale(map_in.getFloat(f64, "scaleY") orelse d.scale_x);
    if (!std.math.isFinite(d.scale_x) or !std.math.isFinite(d.scale_y) or d.scale_x <= 0 or d.scale_y <= 0) {
        map_out.setError("ScaleVect requires finite scale factors greater than zero.");
        vsapi.?.freeNode.?(d.node);
        return;
    }
    const peek = vsapi.?.getFrame.?(0, d.node, null, 0) orelse {
        map_out.setError("ScaleVect could not retrieve frame 0 to inspect analysis metadata.");
        vsapi.?.freeNode.?(d.node);
        return;
    };
    defer vsapi.?.freeFrame.?(peek);
    const props = zapi.initZMap(vsapi.?.getFramePropertiesRO.?(peek));
    const analysis_bytes = props.getData("MVTools_MVAnalysisData", 0) orelse {
        map_out.setError("ScaleVect could not read MVTools_MVAnalysisData from frame 0.");
        vsapi.?.freeNode.?(d.node);
        return;
    };
    const analysis = readAnalysisData(analysis_bytes) orelse {
        map_out.setError("ScaleVect found an invalid MVTools_MVAnalysisData property on frame 0.");
        vsapi.?.freeNode.?(d.node);
        return;
    };
    if (analysis.block_size_x <= analysis.overlap_x or analysis.block_size_y <= analysis.overlap_y or analysis.block_count_x == 0 or analysis.block_count_y == 0 or analysis.level_count == 0) {
        map_out.setError("ScaleVect found malformed block-grid metadata on frame 0.");
        vsapi.?.freeNode.?(d.node);
        return;
    }
    d.geometry.x = makeAxisGeometry(analysis.width, analysis.block_size_x, analysis.overlap_x, analysis.padding_x, analysis.block_count_x, analysis.level_count, d.scale_x) orelse {
        map_out.setError("ScaleVect could not represent the scaled horizontal geometry safely.");
        vsapi.?.freeNode.?(d.node);
        return;
    };
    d.geometry.y = makeAxisGeometry(analysis.height, analysis.block_size_y, analysis.overlap_y, analysis.padding_y, analysis.block_count_y, analysis.level_count, d.scale_y) orelse {
        map_out.setError("ScaleVect could not represent the scaled vertical geometry safely.");
        vsapi.?.freeNode.?(d.node);
        return;
    };
    const original_area = @as(f64, @floatFromInt(analysis.block_size_x)) * @as(f64, @floatFromInt(analysis.block_size_y));
    const scaled_area = @as(f64, @floatFromInt(d.geometry.x.block_size)) * @as(f64, @floatFromInt(d.geometry.y.block_size));
    d.sad_scale = scaled_area / original_area;
    const compatibility_issues = collectCompatibilityIssues(analysis, d.geometry);
    const data: *FunctionData = allocator.create(FunctionData) catch {
        map_out.setError("ScaleVect could not allocate filter state.");
        vsapi.?.freeNode.?(d.node);
        return;
    };
    data.* = d;
    logCompatibilityWarning(compatibility_issues, core, vsapi);
    var deps = [_]vs.FilterDependency{
        vs.FilterDependency{
            .source = d.node,
            .requestPattern = .General,
        },
    };
    vsapi.?.createVideoFilter.?(out, "ScaleVect", d.vi, getFrameScaleVect, freeScaleVect, .Parallel, &deps, deps.len, data, core);
}
