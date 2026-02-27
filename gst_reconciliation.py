import networkx as nx
from datetime import datetime
from typing import Dict, List, Tuple

class GSTKnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
    
    def add_invoice(self, invoice_id: str, vendor: str, amount: float, gst_rate: float, date: str):
        """Add invoice node and relationships"""
        gst_amount = amount * gst_rate / 100
        
        self.graph.add_node(invoice_id, type='invoice', amount=amount, gst=gst_amount, date=date)
        self.graph.add_node(vendor, type='vendor')
        self.graph.add_node(f"GST_{gst_rate}", type='gst_rate', rate=gst_rate)
        
        self.graph.add_edge(invoice_id, vendor, relation='issued_by')
        self.graph.add_edge(invoice_id, f"GST_{gst_rate}", relation='taxed_at')
    
    def add_gstr_entry(self, entry_id: str, invoice_ref: str, amount: float, gst: float):
        """Add GSTR filing entry"""
        self.graph.add_node(entry_id, type='gstr_entry', amount=amount, gst=gst)
        self.graph.add_edge(entry_id, invoice_ref, relation='references')
    
    def reconcile(self) -> List[Dict]:
        """Find mismatches between invoices and GSTR entries"""
        issues = []
        
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'invoice':
                gstr_entries = [n for n in self.graph.predecessors(node) 
                               if self.graph.nodes[n].get('type') == 'gstr_entry']
                
                if not gstr_entries:
                    issues.append({'type': 'missing_gstr', 'invoice': node, 'amount': data['amount']})
                else:
                    for entry in gstr_entries:
                        entry_data = self.graph.nodes[entry]
                        if abs(data['gst'] - entry_data['gst']) > 0.01:
                            issues.append({
                                'type': 'gst_mismatch',
                                'invoice': node,
                                'expected': data['gst'],
                                'actual': entry_data['gst'],
                                'difference': data['gst'] - entry_data['gst']
                            })
        
        return issues
    
    def vendor_summary(self, vendor: str) -> Dict:
        """Get vendor-wise GST summary"""
        invoices = [n for n in self.graph.predecessors(vendor) 
                   if self.graph.nodes[n].get('type') == 'invoice']
        
        total_amount = sum(self.graph.nodes[inv]['amount'] for inv in invoices)
        total_gst = sum(self.graph.nodes[inv]['gst'] for inv in invoices)
        
        return {'vendor': vendor, 'invoices': len(invoices), 'total_amount': total_amount, 'total_gst': total_gst}
    
    def query_path(self, start: str, end: str) -> List:
        """Find relationship path between entities"""
        try:
            return nx.shortest_path(self.graph, start, end)
        except nx.NetworkXNoPath:
            return []

# Example Usage
if __name__ == "__main__":
    kg = GSTKnowledgeGraph()
    
    # Add invoices
    kg.add_invoice("INV001", "Vendor_A", 10000, 18, "2024-01-15")
    kg.add_invoice("INV002", "Vendor_B", 5000, 12, "2024-01-20")
    kg.add_invoice("INV003", "Vendor_A", 8000, 18, "2024-01-25")
    
    # Add GSTR entries
    kg.add_gstr_entry("GSTR001", "INV001", 10000, 1800)
    kg.add_gstr_entry("GSTR002", "INV002", 5000, 500)  # Mismatch
    
    # Reconcile
    issues = kg.reconcile()
    print("Reconciliation Issues:")
    for issue in issues:
        print(f"  {issue}")
    
    # Vendor summary
    print("\nVendor Summary:")
    print(kg.vendor_summary("Vendor_A"))
