import networkx as nx
import json
import os
import requests
from typing import List, Dict

class GraphEngine:
    """
    Manages Knowledge Graph extraction and persistence for GraphRAG.
    """
    def __init__(self, graph_path: str = "knowledge_graph.gml", ollama_url: str = "http://localhost:11434"):
        self.graph_path = graph_path
        self.ollama_url = ollama_url
        self.graph = nx.MultiDiGraph()
        
        if os.path.exists(graph_path):
            try:
                self.graph = nx.read_gml(graph_path)
                print(f"📂 Loaded Knowledge Graph with {self.graph.number_of_nodes()} nodes")
            except:
                print("⚠️ Could not load graph, initializing new one")

    def save_graph(self):
        nx.write_gml(self.graph, self.graph_path)

    def extract_triplets(self, text: str, model: str = "llama3:latest") -> List[Dict]:
        """
        Extract (subject, predicate, object) triplets from text using LLM.
        """
        prompt = f"""
        Extract at most 5 key knowledge triplets from the text below. 
        Format as JSON list: [{{"subject": "...", "predicate": "...", "object": "..."}}]
        
        Text: {text}
        
        JSON:
        """
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json=data, timeout=120)
            if response.status_code == 200:
                result = json.loads(response.json().get('response', '[]'))
                return result if isinstance(result, list) else []
        except Exception as e:
            print(f"⚠️ Triplet extraction failed: {e}")
        return []

    def add_to_graph(self, triplets: List[Dict], source: str):
        """Add extracted triplets to the networkx graph."""
        for t in triplets:
            sub = t.get('subject', '').strip()
            pred = t.get('predicate', '').strip()
            obj = t.get('object', '').strip()
            
            if sub and pred and obj:
                self.graph.add_edge(sub, obj, relation=pred, source=source)
        self.save_graph()

    def search_relationships(self, entity: str) -> List[str]:
        """Find related nodes for a given entity."""
        if not self.graph.has_node(entity):
            return []
            
        related = []
        for neighbor in self.graph.neighbors(entity):
            edge_data = self.graph.get_edge_data(entity, neighbor)
            for key in edge_data:
                rel = edge_data[key].get('relation', 'connected to')
                related.append(f"{entity} {rel} {neighbor}")
        return related
