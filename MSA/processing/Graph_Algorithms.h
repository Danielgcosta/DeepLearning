#ifndef GRAPH_ALGORITHMS_H
#define GRAPH_ALGORITHMS_H

#include <vector>
#include <deque>
#include <math.h>
#include <limits>

namespace MSA
{

/**
  Breadth-first search is one of the simplest algorithms for searching a graph and the archetype for many important graph algorithms. Prim's minimum-spanning-tree
algorithm and Dijkstra's single-source shortest-paths algorithm use ideas similar to those in breadth-first search. Given a graph G = (V, E)
and a distinguished source vertex s, breadth-first search systematically explores the edges of G to "discover" every vertex that is reachable from s. It computes
the distance (smallest number of edges) from s to each reachable vertex. It also produces a "breadth-first tree" with root s that contains all reachable vertices.
For any vertex v reachable from s, the path in the breadth-first tree from s to v corresponds to a "shortest path" from s to v in G, that is, a path containing
the smallest number of edges. The algorithm works on both directed and undirected graphs. Breadth-first search is so named because it expands the frontier
between discovered and undiscovered vertices uniformly across the breadth of the frontier. That is, the algorithm discovers all vertices at distance k from s
before discovering any vertices at distance k + 1.
  To keep track of progress, the method search colors each vertex white, gray, or black. All vertices start out white and may later become gray and then black.
A vertex is discovered the first time it is encountered during the search, at which time it becomes nonwhite.
Gray and black vertices, therefore, have been discovered, but breadth-first search distinguishes between them to ensure that the search proceeds in a breadth-first
manner. If (u, v) pert. E and vertex u is black, then vertex v is either gray or black; that is, all vertices adjacent to black vertices have been discovered.
Gray vertices may have some adjacent white vertices; they represent the frontier between discovered and undiscovered vertices. Breadth-first search constructs a
breadth-first tree, initially containing only its root, which is the source vertex s. Whenever a white vertex v is discovered in the course of scanning the
adjacency list of an already discovered vertex u, the vertex v and the edge (u, v) are added to the tree. We say that u is the predecessor or parent of v in the
breadth-first tree. Since a vertex is discovered at most once, it has at most one parent. Ancestor and descendant relationships in the breadth-first tree are
defined relative to the root s as usual: if u is on a path in the tree from the root s to vertex v, then u is an ancestor of v and v is a descendant of u. The
breadth-first-search procedure BFS below assumes that the input graph G = (V, E) is represented using adjacency lists. It maintains several additional data
structures with each vertex in the graph. The color of each vertex u V is stored in the variable color[u], and the predecessor of u is stored in the variable
π[u]. If u has no predecessor (for example, if u = s or u has not been discovered), then π[u] = NIL. The distance from the source s to vertex u computed by
the algorithm is stored in d[u]. The algorithm also uses a first-in, first-out queue Q to manage the set of gray vertices.


Variáveis:
G   -   Grafo;
V   -   Lista (vetor, matriz, etc) de vértices;
E   -   Lista (vetor, matriz, etc) de arestas;
v   -   Vértice qualquer do grafo;
s   -   Fonte a partir da qual executar a busca;

Cada vértice u possui as seguintes propriedades:
  color[u]    -   white, gray, ou black;
  d[u]        -   distância do vértice s ao vértice u;
  π[u]        -   informa o predecessor (pai) do vértice u.

O algoritmo de busca em largura:
BFS(G, s)
  for each vertex u pert. [G] - {s}
    do color[u] ← WHITE
    d[u] ← ∞
    π[u] ← NIL
  color[s] ← GRAY
  d[s] ← 0
  π[s] ← NIL
  Q ← Ø
  ENQUEUE(Q, s)
  while Q ≠ Ø
    do u ← DEQUEUE(Q)
      for each v pert Adj[u]
        do if color[v] = WHITE
          then color[v] ← GRAY
            d[v] ← d[u] + 1
            π[v] ← u
            ENQUEUE(Q, v)
      color[u] ← BLACK
*/



/**
  * Classe AUXILIAR que representa um nó de um grafo.
  */
class graphNode
{
public:

  // Define a cor de um nó:
  enum nodeColor
  {
    white,
    gray,
    black
  };

  nodeColor gnColor;
  int distanceFromS;
  int predecessorId;
  int nodeId;

  /**
    * Construtor.
    */
  graphNode( int nId=0 )
  {
    gnColor = white;
    distanceFromS = std::numeric_limits<int>::max();
    predecessorId = -1;
    nodeId = nId;
  }

  /**
    * Destrutor.
    */
  ~graphNode()
  {
  }
};


/**
  * Classe que representa um grafo.
  */
class Graph
{
public:
  std::vector<graphNode*>  _graphNodes;
  std::vector<std::vector<int> > _edgesMatrix;

  /**
    * Construtor. Recebe o vetor de arestas.
    */
  Graph( std::vector<std::vector<int> >& edgesMatrix )
  {
    // Recebe o vetor de arestas e cria o vetor de vértices, já instanciando cada vértice:
    _edgesMatrix = edgesMatrix;
    _graphNodes.reserve( _edgesMatrix.size() );
    _graphNodes.resize(  _edgesMatrix.size() );
    for( unsigned int i=0 ; i<_edgesMatrix.size() ; i++ )
    {
      graphNode* thisNode = new graphNode();
      _graphNodes[i] = thisNode;
    }
  }


  /**
    * Destrutor.
    */
  ~Graph()
  {
    _graphNodes.clear();
  }


  /**
    * Breadth-first search:
    * Busca em largura.
    * @param sId Id do vértice a partir do qual se deseja encontrar as distâncias a todos os outros (em número de arestas).
    * @param distanceFromS Vetor contendo as distâncias, a ser retornado.
    * @return Retorna -1 em caso de erro, ou 0 caso ok.
    */
  int BFS( int sId, std::vector<int>& distancesFromS )
  {
    if( sId >= (int)_edgesMatrix.size() )
      return -1;

    if( _edgesMatrix.size() != _graphNodes.size() )
      return -1;

    // Alocando espaco para o vetor de distâncias a ser retornado:
    distancesFromS.reserve( _edgesMatrix.size() );
    distancesFromS.resize(  _edgesMatrix.size() );

    // for each vertex u pert. [G] - {s}, color[u] ← WHITE, d[u] ← ∞, π[u] ← NIL:
    for( unsigned int i=0 ; i<_graphNodes.size() ; i++ )
    {
      graphNode* thisNode = _graphNodes[i];
      thisNode->gnColor = graphNode::white;
      thisNode->distanceFromS = std::numeric_limits<int>::max();
      thisNode->predecessorId = -1;
      thisNode->nodeId = i;
    }

    // color[s] ← GRAY, d[s] ← 0, π[s] ← NIL
    graphNode* s = _graphNodes[sId];
    s->gnColor = graphNode::gray;
    s->distanceFromS = 0;
    s->predecessorId = -1;

    // Q ← Ø:
    std::deque<graphNode*> Q;
    // ENQUEUE(Q, s)
    Q.push_back(s);
    unsigned long firstPos = 0;
    // while Q ≠ Ø:
    while( firstPos < Q.size() )
    {
      graphNode* u = Q[firstPos];
      firstPos++;
      int uId = u->nodeId;
      // for each v pert Adj[u]:
      for( int i=0 ; i<(int)_edgesMatrix.size() ; i++ )
      {
        if( i == uId ) continue;
        if( _edgesMatrix[uId][i] == 0 ) continue;

        // do if color[v] = WHITE:
        graphNode* v = _graphNodes[i];
        if( v->gnColor == graphNode::white )
        {
          // color[v] ← GRAY, d[v] ← d[u] + 1, π[v] ← u, ENQUEUE(Q, v):
          v->gnColor = graphNode::gray;
          v->distanceFromS = u->distanceFromS + 1;
          v->predecessorId = u->nodeId;
          Q.push_back(v);
        }
      }
      // color[u] ← BLACK:
      u->gnColor = graphNode::black;
    }

    // Copiando as distâncias no vetor e retornando:
    for( unsigned int i=0 ; i<_graphNodes.size() ; i++ )
    {
      graphNode* thisNode = _graphNodes[i];
      distancesFromS[i] = thisNode->distanceFromS;
    }
    return 0;
  }


};

}

#endif // GRAPH_ALGORITHMS_H
