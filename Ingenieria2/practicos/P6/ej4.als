
sig Node {}

// no lo voy a usar
sig Edge {
    from, to: Node
}

sig Graph {
    nodes: set Node,
    edges: nodes -> nodes,
}{
    #nodes >= 1
    not no nodes
}

fun idenNodos[]: (Node -> Node){
	iden & (Node -> Node)
}

fun univEdges[]: (Node -> Node){
	(Node -> Node)
}

fun undirected[g: Graph]: Node -> Node {
    g.edges + ~(g.edges)
}

// (a) el grafo es acıclico,

// pred aciclicoBad[g: Graph]{
// 	no (^(g.edges) & ~(g.edges))
// }

pred aciclico2[g: Graph]{
	no (^(g.edges) & idenNodos)
}

run aciclico for 4 but exactly 1 Graph
// run {pred} for {numero maximo de sig creadas} but {cosas que quiero de cada signature}
// check grafos for 5 but exactly 1 Grafo

// ^ es la estrllita
// un facto si quisiera que todos mis grados sean aciclicos
// fact {
//     all no (^(g.aristas) & ~(g.aristas))
// }

// (b) el grafo es no dirigido,

pred noDirigido[g: Graph]{
    g.edges = ~(g.edges)
}

// (c) el grafo es fuertemente conexo,

pred strConecedt[g: Graph]{
    ^(g.edges) + idenNodos = univNodos  
}

pred strConected2(g: Graph) {
    all n1, n2: g.nodes | n1 -> n2 in ^(g.edges)
}

// (d) el grafo es conexo,

pred wklyConectedBAD(g:Graph){
    all n1, n2: g.nodes | (n1 -> n2 in ^(g.edges) || n2 -> n1 in ^(g.edges))
}

pred wklyConected(g:Graph){
    all n1, n2: g.nodes | n1 -> n2 in ^(g.edges + ~(g.edges))
}

pred wklyConectedWFunction[g: Graph] {
    all n1, n2: g.nodes | n2 in ^(undirected[g])[n1]
}

pred conexo[g: Graph]{
	^(g.edges + ~(g.edges)) = univEdges or #g.nodes = 1
}

// (e) el grafo contiene una componente fuertemente conexa,

pred componenteStrongConexo[g: Graph]{
	some disj n1, n2: N |
		(n1 -> n2) in ^(g.edges) and
		(n2 -> n1) in ^(g.edges)
}

pred stgConectedComponent[g: Graph] {
    some s: set g.nodes |
        #s >= 2 and
        all n1, n2: s |
            (n1 -> n2) in ^(g.edges) and
		    (n2 -> n1) in ^(g.edges)
}

// con funcion auxiliar

pred fuertementeConexoAux[g: Graph, s: set Node] {
    all n1, n2: s |
        n2 in ^(g.edges)[n1] and
        n1 in ^(g.edges)[n2]
}

pred stgConectedComponent2[g: Graph] {
    some s: set g.nodes |
        #s >= 2 and
        all n1, n2: s |
            fuertementeConexoAux[g, s]
}


// (f) el grafo contiene una componente conexa,

pred conectedComponent[g: Graph]{
    some s: set g.nodes |
        #s >= 2 and
        all n1, n2: s |
            n1 -> n2 in ^(g.edges + ~(g.edges))
}