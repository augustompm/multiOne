from graphviz import Digraph

def create_multi_adaptive_pam_flowchart():
    # Create Digraph object
    dot = Digraph('multi_adaptive_pam_flow')
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

    # Main nodes
    dot.node('start', 'Start', 
            shape='oval',
            fillcolor='#E8F0FE')
    
    dot.node('init_matrices', 'Initialize Three PAM250-based\nMatrices (HIGH/MEDIUM/LOW)')
    
    dot.node('analyze_blocks', 'Analyze Reference Alignment\nBlocks & Conservation Levels')
    
    dot.node('init_pop', 'Initialize Hierarchical Population\nwith Multi-Matrix Individuals')
    
    dot.node('genetic_ops', 'Perform Genetic Operators\n(Hierarchical Crossover & Mutation)')
    
    dot.node('local_search', 'Apply VNS-ILS Local Search\nPer Conservation Level')
    
    dot.node('eval', 'Evaluate Population Using\nLevel-Specific Scoring')
    
    dot.node('check_stop', 'Check Stop\nConditions',
            shape='diamond',
            fillcolor='#FFE4E1',
            width='1.5',
            height='1.5')
    
    dot.node('end', 'Return Best Matrix Set',
            shape='oval',
            fillcolor='#E8F0FE')

    # Information nodes
    dot.node('conservation_info',
            'Conservation Levels:\n' +
            '• HIGH: >25.0\n' +
            '• MEDIUM: 20.0-25.0\n' +
            '• LOW: <20.0',
            shape='note',
            style='filled',
            fillcolor='#F0F8FF')
    
    dot.node('stop_conditions',
            'Stop Conditions:\n' +
            '• Max generations (50)\n' +
            '• Stagnation (10 gen)\n' +
            '• Time limit (2h)\n' +
            '• Population converged',
            shape='note',
            style='filled',
            fillcolor='#F0F8FF')

    # Main flow
    dot.edge('start', 'init_matrices')
    dot.edge('init_matrices', 'analyze_blocks')
    dot.edge('analyze_blocks', 'init_pop')
    dot.edge('init_pop', 'genetic_ops')
    dot.edge('genetic_ops', 'local_search')
    dot.edge('local_search', 'eval')
    dot.edge('eval', 'check_stop')
    
    # Feedback loop and end
    dot.edge('check_stop', 'genetic_ops',
            xlabel='No',
            constraint='false')
    dot.edge('check_stop', 'end',
            xlabel='Yes')

    # Connect information nodes
    dot.edge('conservation_info', 'analyze_blocks',
            style='dotted',
            arrowhead='none',
            constraint='false')
    
    dot.edge('stop_conditions', 'check_stop',
            style='dotted',
            arrowhead='none',
            constraint='false')

    # Align information nodes
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('analyze_blocks')
        s.node('conservation_info')
    
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('check_stop')
        s.node('stop_conditions')

    dot.attr(bgcolor='transparent')
    
    return dot

if __name__ == '__main__':
    flow = create_multi_adaptive_pam_flowchart()
    
    # Save with high quality
    flow.render('multi_adaptive_pam_flowchart',
               format='pdf',
               cleanup=True)
               
    flow.render('multi_adaptive_pam_flowchart',
               format='png',
               cleanup=True)