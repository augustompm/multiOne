# Análise Abrangente de Alinhamentos Múltiplos de Sequências

## Justificativa para Abordagem Multi-objetivo

Este relatório apresenta evidências quantitativas e qualitativas que suportam o desenvolvimento de uma abordagem multi-objetivo para alinhamento múltiplo de sequências, com foco especial em conservação estrutural.

### 1. Estatísticas por Método


#### ClustalW
- SP Mean: 0.1710
- SP Std: 0.0860
- SP Median: 0.1628
- WSP Mean: 0.8153
- WSP Std: 0.0977
- WSP Median: 0.8093
- SP-WSP Correlation: -0.4903
- Sample Size: 60
- SP Normality p-value: 1.8500e-01
- WSP Normality p-value: 5.8259e-01

**Interpretação:**
- Performance Local (SP): Baixo nível de conservação local.
- Performance Global (WSP): Alto nível de conservação global.
- Trade-off: Moderado trade-off entre conservação local e global.

#### MUSCLE
- SP Mean: 0.1489
- SP Std: 0.0814
- SP Median: 0.1387
- WSP Mean: 0.8738
- WSP Std: 0.0960
- WSP Median: 0.8762
- SP-WSP Correlation: -0.6973
- Sample Size: 60
- SP Normality p-value: 2.1428e-01
- WSP Normality p-value: 6.8026e-01

**Interpretação:**
- Performance Local (SP): Baixo nível de conservação local.
- Performance Global (WSP): Alto nível de conservação global.
- Trade-off: Moderado trade-off entre conservação local e global.

#### BAliBASE
- SP Mean: 0.1676
- SP Std: 0.0857
- SP Median: 0.1674
- WSP Mean: 0.8157
- WSP Std: 0.0956
- WSP Median: 0.8125
- SP-WSP Correlation: -0.5550
- Sample Size: 60
- SP Normality p-value: 2.2607e-01
- WSP Normality p-value: 5.0285e-01

**Interpretação:**
- Performance Local (SP): Baixo nível de conservação local.
- Performance Global (WSP): Alto nível de conservação global.
- Trade-off: Moderado trade-off entre conservação local e global.

### 2. Análise Estatística Avançada

#### Teste de Kruskal-Wallis
**SP Scores:**
- H-statistic: 2.3647
- p-value: 3.0656e-01
**WSP Scores:**
- H-statistic: 13.2886
- p-value: 1.3014e-03

#### Análise de Correlação

**ClustalW:**
- Correlação de Spearman: -0.5355
- p-value: 1.0406e-05
- A correlação é significativa ao nível de 5%.

**MUSCLE:**
- Correlação de Spearman: -0.7573
- p-value: 2.5290e-12
- A correlação é significativa ao nível de 5%.

**BAliBASE:**
- Correlação de Spearman: -0.6193
- p-value: 1.3284e-07
- A correlação é significativa ao nível de 5%.

### 3. Recomendações

- Considerar uma abordagem multi-objetivo que balanceie a conservação local e global para otimizar a qualidade dos alinhamentos.
- Métodos que demonstram forte correlação negativa entre SP e WSP indicam a necessidade de otimização para mitigar o trade-off observado.
- Avaliar a normalidade dos dados para selecionar testes estatísticos apropriados nas análises futuras.