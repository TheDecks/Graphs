from kegg.pathway import Pathway


hsa00010_pathway = Pathway.from_url("http://rest.kegg.jp/get/hsa00010/kgml")

root = hsa00010_pathway.root_graph
for node in root.nodes:
    edges_from = root.edges_from(node)
    if edges_from:
        print(node)
        print('\n\t'.join(str(edge) for edge in edges_from))

print('-----------------------------------------------')
m_node = max(root.nodes, key=lambda x: root.degree_out(x))
print(m_node, root.edges_from(m_node))
