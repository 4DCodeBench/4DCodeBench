// scorer-vdb: dense velocity grids and mesh level sets for the scorer's liquid sampler.
//   velocity IN OUT nx ny nz   the `velocity` Vec3S grid of a Mantaflow VDB file
//   surface  IN OUT nx ny nz   the narrow-band signed distance, in voxels, of a binary mesh
//                              (int32 V, T; V float32 vertices; T int32 triangles)
// OUT is the grid over [0, n) as raw float32 in C order over (nx, ny, nz).
#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include <openvdb/tools/MeshToVolume.h>
#include <tbb/global_control.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

template<class T> void read(std::istream& in, T* data, size_t count) {
    in.read(reinterpret_cast<char*>(data), count * sizeof(T));
    if (!in) throw std::runtime_error("truncated mesh input");
}

template<class Grid> void dump(const Grid& grid, openvdb::Coord dim, const char* path) {
    using Value = typename Grid::ValueType;
    std::vector<Value> values(size_t(dim.x()) * dim.y() * dim.z());
    openvdb::tools::Dense<Value> dense(openvdb::CoordBBox(openvdb::Coord(0), dim - openvdb::Coord(1)), values.data());
    openvdb::tools::copyToDense(grid, dense);
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(Value));
    if (!out) throw std::runtime_error("cannot write dense grid");
}

int main(int argc, char** argv) {
    try {
        if (argc != 7) throw std::runtime_error("usage: scorer-vdb velocity|surface input output nx ny nz");
        const char* threads = std::getenv("OMP_NUM_THREADS");
        tbb::global_control limit(tbb::global_control::max_allowed_parallelism,
                                  threads ? std::stoi(threads) : 1);
        openvdb::initialize();
        openvdb::Coord dim(std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[6]));
        const std::string mode(argv[1]);
        if (mode == "velocity") {
            openvdb::io::File file(argv[2]);
            file.open();
            auto grid = openvdb::gridPtrCast<openvdb::Vec3SGrid>(file.readGrid("velocity"));
            if (!grid) throw std::runtime_error("velocity must be a Vec3S grid");
            auto res = grid->metaValue<openvdb::Vec3i>("file_base_resolution");
            if (res != dim.asVec3i()) throw std::runtime_error("velocity resolution disagrees with configuration");
            dump(*grid, dim, argv[3]);
        } else if (mode == "surface") {
            std::ifstream in(argv[2], std::ios::binary);
            int32_t counts[2];
            read(in, counts, 2);
            std::vector<openvdb::Vec3s> points(counts[0]);
            std::vector<openvdb::Vec3I> triangles(counts[1]);
            read(in, points.data(), points.size());
            read(in, triangles.data(), triangles.size());
            auto grid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(
                *openvdb::math::Transform::createLinearTransform(), points, triangles);
            dump(*grid, dim, argv[3]);
        } else {
            throw std::runtime_error("unknown grid operation: " + mode);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
