from graphviz import Digraph

def create_adaptive_pam_flowchart():
    # Create Digraph object
    dot = Digraph('adaptive_pam_flow')
    dot.attr(rankdir='TB')
    
    # Global graph attributes
    dot.attr('graph',
            nodesep='0.5',
            ranksep='0.5',
            splines='spline',
            center='true')
    
    # Node attributes
    dot.attr('node',
            shape='rectangle',
            style='rounded,filled',
            fillcolor='white',
            fontname='Arial',
            fontsize='11',
            width='2.5',
            height='0.6',
            margin='0.2')
    
    # Edge attributes
    dot.attr('edge',
            fontname='Arial',
            fontsize='10',
            arrowsize='0.8')

    # Nodes
    dot.node('start', 'Start',
            shape='oval',
            fillcolor='#E8F0FE')
    
    dot.node('init_pop', 'Initialize Population\nwith PAM250-based Matrices')
    dot.node('ga_ops', 'Perform Genetic Algorithm\nOperations')
    dot.node('local_search', 'Apply Local Search\nto Individuals')
    dot.node('eval_pop', 'Evaluate Population\nFitness')
            
    dot.node('check_stop', 'Check Stop\nConditions',
            shape='diamond',
            fillcolor='#FFE4E1',
            width='1.5',
            height='1.5')
    
    dot.node('end', 'Return Best Matrix',
            shape='oval',
            fillcolor='#E8F0FE')

    dot.node('stop_conditions',
            'Stop Conditions:\n' +
            '• Max generations reached\n' +
            '• Stagnation limit\n' +
            '• Time limit exceeded\n' +
            '• Population converged',
            shape='note',
            style='filled',
            fillcolor='#F0F8FF')

    # Main flow in correct order
    dot.edge('start', 'init_pop')
    dot.edge('init_pop', 'ga_ops')
    dot.edge('ga_ops', 'local_search')
    dot.edge('local_search', 'eval_pop')
    dot.edge('eval_pop', 'check_stop')
    
    # Feedback loop and final edge
    dot.edge('check_stop', 'ga_ops',
            xlabel='No',
            tailport='w',
            headport='e',
            constraint='false')
            
    dot.edge('check_stop', 'end',
            xlabel='Yes',
            tailport='s')

    # Connect stop conditions
    dot.edge('stop_conditions', 'check_stop',
            style='dotted',
            arrowhead='none',
            constraint='false')

    # Align stop conditions
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('check_stop')
        s.node('stop_conditions')

    dot.attr(bgcolor='transparent')
    
    return dot

if __name__ == '__main__':
    flow = create_adaptive_pam_flowchart()
    
    # Save with high quality
    flow.render('adaptive_pam_flowchart',
               format='pdf',
               cleanup=True)
               
    flow.render('adaptive_pam_flowchart',
               format='png',
               cleanup=True)