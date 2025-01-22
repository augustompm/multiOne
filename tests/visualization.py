import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import sys
import os
import matplotlib.patches as patches
import matplotlib.transforms as transforms

class StructuralAlignmentAnalyzer:
    """
    Analisador abrangente de alinhamentos múltiplos de sequências com foco em
    padrões estruturais e justificativa para otimização multi-objetivo.
    
    Esta classe implementa visualizações e análises que demonstram:
    1. Trade-off entre conservação local (SP) e global (WSP)
    2. Padrões de conservação estrutural
    3. Variabilidade entre diferentes métodos de alinhamento
    4. Justificativa estatística para abordagem multi-objetivo
    """
    
    def __init__(self, csv_path: str):
        """
        Inicializa o analisador com configurações otimizadas para visualização
        e análise estatística.
        
        Parameters:
            csv_path (str): Caminho para arquivo CSV com scores de alinhamento
                          Formato esperado: sequence,method,sp_norm,wsp_norm,...
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Arquivo CSV não encontrado: {self.csv_path}")
            
        # Cria diretório para resultados se não existir
        self.output_dir = Path('structural_analysis_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Carrega dados e configura ambiente visual
        self.data = pd.read_csv(self.csv_path)
        print(f"Análise iniciada com {len(self.data)} alinhamentos")
        
        # Configuração visual moderna usando API do seaborn
        sns.set_theme(style="whitegrid", font_scale=1.2)
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 300
        
        # Validação inicial dos dados
        self._validate_data()

    def _validate_data(self):
        """
        Valida integridade e formato dos dados de entrada.
        Verifica presença de colunas necessárias e valores válidos.
        """
        required_columns = ['sequence', 'method', 'sp_norm', 'wsp_norm']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError("Dados não contêm todas as colunas necessárias")
        
        # Opcional: Verificar se existem valores faltantes
        if self.data[required_columns].isnull().any().any():
            raise ValueError("Existem valores faltantes nas colunas necessárias")

    def compute_statistics(self):
        """
        Calcula estatísticas abrangentes para cada método de alinhamento.
        Inclui médias, desvios, medianas e correlações entre SP e WSP.
        
        Returns:
            pd.DataFrame: DataFrame com estatísticas por método
        """
        stats_dict = {}
        methods = self.data['method'].unique()
        
        for method in methods:
            method_data = self.data[self.data['method'] == method]
            
            # Cálculo de estatísticas básicas e correlações
            stats_dict[method] = {
                'SP Mean': f"{method_data['sp_norm'].mean():.4f}",
                'SP Std': f"{method_data['sp_norm'].std():.4f}",
                'SP Median': f"{method_data['sp_norm'].median():.4f}",
                'WSP Mean': f"{method_data['wsp_norm'].mean():.4f}",
                'WSP Std': f"{method_data['wsp_norm'].std():.4f}",
                'WSP Median': f"{method_data['wsp_norm'].median():.4f}",
                'SP-WSP Correlation': f"{method_data['sp_norm'].corr(method_data['wsp_norm']):.4f}",
                'Sample Size': len(method_data)
            }
            
            # Adiciona testes de normalidade
            _, sp_norm_p = stats.normaltest(method_data['sp_norm'])
            _, wsp_norm_p = stats.normaltest(method_data['wsp_norm'])
            stats_dict[method]['SP Normality p-value'] = f"{sp_norm_p:.4e}"
            stats_dict[method]['WSP Normality p-value'] = f"{wsp_norm_p:.4e}"
        
        return pd.DataFrame(stats_dict).T

    def plot_score_distribution(self):
        """
        Cria visualização da distribuição conjunta de scores SP e WSP,
        destacando o trade-off entre conservação local e global.
        """
        plt.figure(figsize=(15, 10))
        
        # Plot principal
        for method in self.data['method'].unique():
            method_data = self.data[self.data['method'] == method]
            
            # Scatter plot com densidade
            sns.scatterplot(data=method_data, 
                          x='sp_norm', y='wsp_norm',
                          label=method, alpha=0.6, s=100)
            
            # Adiciona elipse de confiança
            confidence_ellipse(method_data['sp_norm'].values,
                             method_data['wsp_norm'].values,
                             plt.gca(), n_std=2.0,
                             label=f'{method} (95% IC)',
                             alpha=0.1)
            
            # Linha de tendência com intervalo de confiança
            sns.regplot(data=method_data,
                       x='sp_norm', y='wsp_norm',
                       scatter=False, color='gray',
                       line_kws={'alpha':0.5})
        
        # Anotações explicativas
        plt.annotate('Alta Conservação Global\nPossível Conservação Estrutural',
                    xy=(0.15, 0.9), xycoords='data',
                    bbox=dict(boxstyle='round,pad=0.5', 
                             fc='yellow', alpha=0.3))
        
        plt.annotate('Baixa Conservação Global\nPossível Perda Estrutural',
                    xy=(0.15, 0.6), xycoords='data',
                    bbox=dict(boxstyle='round,pad=0.5', 
                             fc='red', alpha=0.3))
        
        # Configuração do plot
        plt.xlabel('Conservação Local (SP Score)', fontsize=12)
        plt.ylabel('Conservação Global (WSP Score)', fontsize=12)
        plt.title('Trade-off entre Conservação Local e Global\n'
                 'Evidência para Necessidade de Otimização Multi-objetivo',
                 fontsize=14, pad=20)
        
        plt.grid(True, alpha=0.3)
        plt.legend(title='Método de Alinhamento')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_distribution.png')
        plt.close()

    def plot_method_comparison(self):
        """
        Cria visualização comparativa entre métodos usando box-plots.
        Destaca distribuição de scores e outliers importantes.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot para SP scores
        sns.boxplot(data=self.data, x='method', y='sp_norm', ax=ax1,
                   width=0.6, palette='husl')
        ax1.set_title('Distribuição de SP Scores por Método\n'
                     'Evidência de Variação na Conservação Local',
                     fontsize=14, pad=20)
        ax1.set_ylabel('SP Score (Normalizado)', fontsize=12)
        ax1.set_xlabel('')
        
        # Adiciona pontos de dados individuais
        sns.stripplot(data=self.data, x='method', y='sp_norm', ax=ax1,
                     color='black', alpha=0.3, size=4)
        
        # Plot para WSP scores
        sns.boxplot(data=self.data, x='method', y='wsp_norm', ax=ax2,
                   width=0.6, palette='husl')
        ax2.set_title('Distribuição de WSP Scores por Método\n'
                     'Indicativo de Conservação Estrutural',
                     fontsize=14, pad=20)
        ax2.set_ylabel('WSP Score (Normalizado)', fontsize=12)
        ax2.set_xlabel('Método de Alinhamento', fontsize=12)
        
        # Adiciona pontos de dados individuais
        sns.stripplot(data=self.data, x='method', y='wsp_norm', ax=ax2,
                     color='black', alpha=0.3, size=4)
        
        # Adiciona linhas de referência
        for ax in [ax1, ax2]:
            ax.axhline(y=0.8, color='g', linestyle='--', alpha=0.3,
                      label='Limiar de Alta Conservação')
            ax.axhline(y=0.6, color='r', linestyle='--', alpha=0.3,
                      label='Limiar de Baixa Conservação')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'method_comparison.png')
        plt.close()

    def plot_sequence_comparison(self):
        """
        Cria visualização da distribuição de scores ao longo das sequências,
        destacando tendências e padrões de conservação.
        """
        plt.figure(figsize=(15, 10))
        
        # Plot principal com tendências
        for method in self.data['method'].unique():
            method_data = self.data[self.data['method'] == method].sort_values('wsp_norm')
            x = range(len(method_data))
            
            # WSP scores
            y_wsp = method_data['wsp_norm']
            plt.plot(x, y_wsp, label=f'{method} WSP',
                    linewidth=2, alpha=0.8)
            
            # Banda de confiança para WSP
            std_wsp = method_data['wsp_norm'].std()
            plt.fill_between(x, y_wsp-std_wsp, y_wsp+std_wsp,
                           alpha=0.2)
            
            # SP scores como comparação
            y_sp = method_data['sp_norm']
            plt.plot(x, y_sp, '--', label=f'{method} SP',
                    alpha=0.5)
        
        # Configuração do plot
        plt.title('Distribuição de Scores por Sequência\n'
                 'Comparação entre Conservação Local (SP) e Global (WSP)',
                 fontsize=14, pad=20)
        plt.xlabel('Índice da Sequência (ordenado por WSP)', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Adiciona anotações explicativas
        plt.annotate('Região de Alta Conservação',
                    xy=(len(x)*0.1, 0.9),
                    xytext=(len(x)*0.2, 0.95),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.annotate('Região de Baixa Conservação',
                    xy=(len(x)*0.8, 0.3),
                    xytext=(len(x)*0.7, 0.2),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sequence_comparison.png',
                   bbox_inches='tight')
        plt.close()

    def plot_method_comparison_advanced(self):
        """
        Cria visualização comparativa entre métodos usando box-plots.
        Destaca distribuição de scores e outliers importantes.
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Prepara dados para visualização pareada
        paired_data = []
        methods = self.data['method'].unique()
        
        for method in methods:
            method_data = self.data[self.data['method'] == method]
            paired_data.append({
                'method': method,
                'sp': method_data['sp_norm'].values,
                'wsp': method_data['wsp_norm'].values
            })
        
        # Cria visualização personalizada
        positions = np.arange(len(methods)) * 3
        colors = sns.color_palette("husl", n_colors=len(methods))
        
        for i, (method_data, color) in enumerate(zip(paired_data, colors)):
            # Violino para SP
            vp = ax.violinplot(method_data['sp'],
                             positions=[positions[i]-0.5],
                             showmeans=True, showextrema=True, showmedians=False)
            for pc in vp['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.3)
            
            # Violino para WSP
            vp = ax.violinplot(method_data['wsp'],
                             positions=[positions[i]+0.5],
                             showmeans=True, showextrema=True, showmedians=False)
            for pc in vp['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            # Adiciona boxplots sobrepostos
            bp = ax.boxplot([method_data['sp'], method_data['wsp']],
                          positions=[positions[i]-0.5, positions[i]+0.5],
                          widths=0.3, showfliers=False, patch_artist=True)
            for box in bp['boxes']:
                box.set(facecolor='none', edgecolor='black')
        
        # Configuração do eixo e labels
        ax.set_xticks(positions)
        ax.set_xticklabels(methods)
        ax.set_title('Comparação Detalhada de Métodos:\n'
                    'SP (claro) vs WSP (escuro) com Distribuição e Quartis',
                    fontsize=14, pad=20)
        
        # Adiciona áreas destacadas para diferentes faixas de performance
        ax.axhspan(0.8, 1.0, color='green', alpha=0.1,
                  label='Região de Alta Conservação')
        ax.axhspan(0.6, 0.8, color='yellow', alpha=0.1,
                  label='Região de Média Conservação')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'method_comparison_advanced.png')
        plt.close()

    def generate_report(self):
        """
        Gera relatório detalhado justificando abordagem multi-objetivo
        baseado nos padrões observados nos dados.
        Inclui análises estatísticas, interpretações e recomendações.
        """
        stats_df = self.compute_statistics()
        
        report = [
            "# Análise Abrangente de Alinhamentos Múltiplos de Sequências",
            "\n## Justificativa para Abordagem Multi-objetivo",
            "\nEste relatório apresenta evidências quantitativas e qualitativas que suportam o desenvolvimento de uma abordagem multi-objetivo para alinhamento múltiplo de sequências, com foco especial em conservação estrutural.",
            
            "\n### 1. Estatísticas por Método\n"
        ]
        
        # Estatísticas básicas e interpretação
        for method in stats_df.index:
            report.append(f"\n#### {method}")
            for stat, value in stats_df.loc[method].items():
                report.append(f"- {stat}: {value}")
            
            # Adiciona interpretação específica
            sp_mean = float(stats_df.loc[method]['SP Mean'])
            wsp_mean = float(stats_df.loc[method]['WSP Mean'])
            corr = float(stats_df.loc[method]['SP-WSP Correlation'])
            
            report.append("\n**Interpretação:**")
            report.append(f"- Performance Local (SP): {self._interpret_sp_score(sp_mean)}")
            report.append(f"- Performance Global (WSP): {self._interpret_wsp_score(wsp_mean)}")
            report.append(f"- Trade-off: {self._interpret_correlation(corr)}")
        
        # Análise estatística avançada
        report.append("\n### 2. Análise Estatística Avançada\n")
        
        # Kruskal-Wallis para diferenças entre métodos
        h_stat_sp, p_value_sp = stats.kruskal(
            *[group['sp_norm'].values for name, group in self.data.groupby('method')]
        )
        h_stat_wsp, p_value_wsp = stats.kruskal(
            *[group['wsp_norm'].values for name, group in self.data.groupby('method')]
        )
        
        report.append("#### Teste de Kruskal-Wallis")
        report.append(f"**SP Scores:**")
        report.append(f"- H-statistic: {h_stat_sp:.4f}")
        report.append(f"- p-value: {p_value_sp:.4e}")
        report.append(f"**WSP Scores:**")
        report.append(f"- H-statistic: {h_stat_wsp:.4f}")
        report.append(f"- p-value: {p_value_wsp:.4e}")
        
        # Análise de correlação
        report.append("\n#### Análise de Correlação")
        for method in stats_df.index:
            method_data = self.data[self.data['method'] == method]
            corr, p_value = stats.spearmanr(method_data['sp_norm'],
                                          method_data['wsp_norm'])
            report.append(f"\n**{method}:**")
            report.append(f"- Correlação de Spearman: {corr:.4f}")
            report.append(f"- p-value: {p_value:.4e}")
            if p_value < 0.05:
                significance = "significativa"
            else:
                significance = "não significativa"
            report.append(f"- A correlação é {significance} ao nível de 5%.")
        
        # Recomendações com base nas análises
        report.append("\n### 3. Recomendações\n")
        report.append("- Considerar uma abordagem multi-objetivo que balanceie a conservação local e global para otimizar a qualidade dos alinhamentos.")
        report.append("- Métodos que demonstram forte correlação negativa entre SP e WSP indicam a necessidade de otimização para mitigar o trade-off observado.")
        report.append("- Avaliar a normalidade dos dados para selecionar testes estatísticos apropriados nas análises futuras.")
        
        # Escrever o relatório em um arquivo Markdown
        report_path = self.output_dir / 'analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"Relatório gerado em: {report_path}")

    def analyze_and_explain(self):
        """
        Executa análise completa e imprime explicações para embasar
        a proposta do SAMOE-MSA.
        """
        print("\nAnálise Estrutural de Alinhamentos Múltiplos")
        print("============================================")
        
        stats = self.compute_statistics()
        
        print("\n1. Evidência de Trade-off Estrutural:")
        for method in stats.index:
            corr = float(stats.loc[method, 'SP-WSP Correlation'])
            print(f"\n{method}:")
            print(f"  Correlação SP-WSP: {corr:.4f}")
            if corr < -0.4:
                print("  → Forte evidência de trade-off entre conservação local e global")
                print("  → Sugere necessidade de otimização multi-objetivo")
            elif corr < -0.2:
                print("  → Moderada evidência de trade-off entre conservação local e global")
            else:
                print("  → Pouca ou nenhuma evidência de trade-off entre conservação local e global")
        
        print("\n2. Padrões de Conservação:")
        for method in stats.index:
            wsp_mean = float(stats.loc[method, 'WSP Mean'])
            sp_mean = float(stats.loc[method, 'SP Mean'])
            print(f"\n{method}:")
            print(f"  WSP Médio: {wsp_mean:.4f}")
            print(f"  SP Médio:  {sp_mean:.4f}")
            print("  → " + self._interpret_scores(wsp_mean, sp_mean))
        
        # Geração de relatórios e plots
        self.generate_report()
        self.plot_score_distribution()
        self.plot_method_comparison()
        self.plot_sequence_comparison()
        self.plot_method_comparison_advanced()
        print("\nVisualizações geradas e salvas no diretório de resultados.")

    def _interpret_sp_score(self, sp):
        """Interpreta os scores SP para justificar a abordagem proposta."""
        if sp > 0.8:
            return "Alto nível de conservação local."
        elif sp > 0.5:
            return "Conservação local moderada."
        else:
            return "Baixo nível de conservação local."

    def _interpret_wsp_score(self, wsp):
        """Interpreta os scores WSP para justificar a abordagem proposta."""
        if wsp > 0.8:
            return "Alto nível de conservação global."
        elif wsp > 0.5:
            return "Conservação global moderada."
        else:
            return "Baixo nível de conservação global."

    def _interpret_correlation(self, corr):
        """Interpreta a correlação entre SP e WSP."""
        if corr < -0.7:
            return "Forte trade-off entre conservação local e global."
        elif corr < -0.4:
            return "Moderado trade-off entre conservação local e global."
        elif corr < -0.2:
            return "Fraco trade-off entre conservação local e global."
        else:
            return "Pouco ou nenhum trade-off entre conservação local e global."

    def _interpret_scores(self, wsp, sp):
        """Interpreta os scores para justificar a abordagem proposta."""
        if wsp > 0.8 and sp < 0.2:
            return "Alta conservação global com baixa local sugere presença de motivos estruturais importantes."
        elif wsp > 0.7 and sp > 0.15:
            return "Bom balanço entre conservação local e global."
        else:
            return "Possível perda de informação estrutural importante."

# Funções auxiliares
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """Adiciona elipse de confiança ao plot."""
    if x.size != y.size:
        raise ValueError("x e y devem ter o mesmo tamanho")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = patches.Ellipse((0, 0),
                               width=ell_radius_x * 2,
                               height=ell_radius_y * 2,
                               facecolor=facecolor,
                               **kwargs)
    
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(np.mean(x), np.mean(y))
    
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


if __name__ == "__main__":
    try:
        csv_path = sys.argv[1] if len(sys.argv) > 1 else 'results/data/alignment_scores-1.csv'
        analyzer = StructuralAlignmentAnalyzer(csv_path)
        analyzer.analyze_and_explain()
    except Exception as e:
        print(f"\nErro: {str(e)}")
        print("\nUso: python structural_visualization.py [caminho_para_csv]")
        sys.exit(1)
