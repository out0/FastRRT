class GraphNode
{
public:
    int x;
    int z;
    float heading_rad{};
    int nodeType;
    int parent_x{};
    int parent_z{};
    float connectToEndCost{};
    float cost{};

    GraphNode(int x, int z, int type) : x(x), z(z), nodeType(type) {}
    GraphNode() : x(0), z(0), heading_rad(0.0f), nodeType(0), parent_x(0), parent_z(0), connectToEndCost(0.0f), cost(0.0f) {}
};