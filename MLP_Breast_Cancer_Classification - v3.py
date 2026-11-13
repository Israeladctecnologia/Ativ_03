

# Importa a biblioteca Pandas, usada para manipulação e análise de dados em formato de tabelas
import pandas as pd
# Importa a função que carrega a base de dados real sobre câncer de mama embutida no Scikit-Learn
from sklearn.datasets import load_breast_cancer
# Importa KFold (para dividir os dados), cross_validate (para rodar a validação e extrair múltiplas métricas)
# e cross_val_predict (para obter a previsão de cada paciente quando ele estava no conjunto de teste)
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
# Importa confusion_matrix (calcula a matriz de confusão) e ConfusionMatrixDisplay (desenha a matriz em gráfico)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Importa o Matplotlib, usado para exibir o gráfico da matriz de confusão
import matplotlib.pyplot as plt
# Importa StandardScaler, responsável por colocar todas as variáveis numéricas na mesma escala (padronização)
from sklearn.preprocessing import StandardScaler
# Importa o MLPClassifier, que é o algoritmo da Rede Neural Artificial (Multi-Layer Perceptron)
from sklearn.neural_network import MLPClassifier
# Importa make_pipeline, que permite encadear os passos de padronização e treinamento de forma segura e automática
from sklearn.pipeline import make_pipeline

# Executa a função para carregar todos os dados reais do dataset de câncer para a variável 'cancer_data'
cancer_data = load_breast_cancer()

# Cria um DataFrame (tabela) do Pandas contendo apenas as características (raio, textura, etc) dos tumores
X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
# Separa a variável alvo (respostas corretas), onde 0 representa Tumor Maligno e 1 representa Tumor Benigno
y = cancer_data.target 

# Imprime na tela a quantidade de pacientes (linhas) e características analisadas (colunas) da nossa base
print(f"Dimensões da base: {X.shape[0]} pacientes e {X.shape[1]} características.")
# Imprime uma linha tracejada apenas para separar visualmente as informações no terminal
print("-" * 50)

# Cria o pipeline que vai garantir a ordem correta de execução (padronizar -> treinar) dentro de cada fold
modelo_pipeline = make_pipeline(
    # Passo 1: Aplica a padronização para que grandezas muito diferentes não confundam a rede neural
    StandardScaler(),
    # Passo 2: Inicializa a Rede Neural com 2 camadas ocultas (30 e 15 neurônios), máximo de 1000 épocas e semente fixa
    MLPClassifier(hidden_layer_sizes=(30, 15), learning_rate_init=0.0001, max_iter=1000, random_state=42)
)

# Configura o K-Fold para dividir a base em 5 partes iguais, embaralhando os pacientes antes da divisão (shuffle=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Mostrando o tamanho das divisões
print("\n--- TAMANHO DAS DIVISÕES (TREINO / TESTE) ---")

# O método kf.split(X) gera os índices (posições) das linhas de treino e teste a cada rodada
for fold, (indices_treino, indices_teste) in enumerate(kf.split(X)):
    qtd_treino = len(indices_treino)
    qtd_teste = len(indices_teste)
    print(f"Fold {fold + 1}: Treino com {qtd_treino} instâncias | Teste com {qtd_teste} instâncias")

print("-" * 50)

# Cria uma lista de textos indicando ao Scikit-Learn quais métricas queremos calcular ao final do processo
metricas = ['accuracy', 'precision', 'recall']

# Imprime um aviso no terminal indicando que o longo processo de treinamento começou
print("\nExecutando o treinamento e validação em 5 Folds...")
# Executa o loop do K-Fold automaticamente: separa os dados, padroniza, treina, testa e salva as métricas
resultados = cross_validate(modelo_pipeline, X, y, cv=kf, scoring=metricas)

# Inicia um laço de repetição (loop) que vai iterar 5 vezes (de 0 até 4) para ler os resultados de cada fold
for i in range(5):
    # Imprime na tela os resultados de acurácia, precisão e recall do fold atual, multiplicando por 100 para virar porcentagem
    print(f"Fold {i+1} | "
          f"Acurácia: {resultados['test_accuracy'][i]*100:.2f}% | "
          f"Precisão: {resultados['test_precision'][i]*100:.2f}% | "
          f"Recall: {resultados['test_recall'][i]*100:.2f}%")

# Imprime outra linha tracejada para separar os resultados de cada fold da conclusão final
print("-" * 50)
# Imprime o cabeçalho anunciando as médias finais
print("\n--- DESEMPENHO MÉDIO FINAL ---")

# Calcula a média (mean) e o desvio padrão (std) da acurácia de todos os 5 testes e imprime em formato percentual
print(f"Acurácia Média: {resultados['test_accuracy'].mean() * 100:.2f}% "
      f"(± {resultados['test_accuracy'].std() * 100:.2f}%)")

# Calcula a média (mean) e o desvio padrão (std) da precisão de todos os 5 testes e imprime em formato percentual
print(f"Precisão Média: {resultados['test_precision'].mean() * 100:.2f}% "
      f"(± {resultados['test_precision'].std() * 100:.2f}%)")

# Calcula a média (mean) e o desvio padrão (std) do recall de todos os 5 testes e imprime em formato percentual
print(f"Recall Médio:   {resultados['test_recall'].mean() * 100:.2f}% "
      f"(± {resultados['test_recall'].std() * 100:.2f}%)")
print("-" * 30)

# ---------------------------------------------------------------------------
# MATRIZ DE CONFUSÃO
# ---------------------------------------------------------------------------

# Gera a previsão de cada paciente usando o mesmo K-Fold: cada paciente é previsto pelo modelo
# treinado nos outros 4 folds, ou seja, nenhuma previsão é feita sobre dados vistos no treino
y_pred = cross_val_predict(modelo_pipeline, X, y, cv=kf)

# Define os nomes das classes na mesma ordem dos rótulos (0 = Maligno, 1 = Benigno)
nomes_classes = ['Maligno', 'Benigno']

# Calcula a matriz de confusão: linhas = classe real, colunas = classe prevista pela rede
matriz = confusion_matrix(y, y_pred, labels=[0, 1])

# Imprime o cabeçalho da seção da matriz de confusão
print("\n--- MATRIZ DE CONFUSÃO (5 FOLDS CONSOLIDADOS) ---")
# Converte a matriz em um DataFrame para imprimir com os nomes das classes nas linhas e colunas
print(pd.DataFrame(matriz,
                   index=[f"Real {c}" for c in nomes_classes],
                   columns=[f"Previsto {c}" for c in nomes_classes]))

# Separa os quatro valores da matriz. A classe positiva do Scikit-Learn é 1 (Benigno),
# a mesma usada no cálculo da precisão e do recall acima
vn, fp, fn, vp = matriz.ravel()

# Imprime a interpretação de cada célula da matriz
print(f"\nMalignos classificados corretamente:          {vn}")
print(f"Malignos classificados como Benignos (erro):  {fp}")
print(f"Benignos classificados como Malignos (erro):  {fn}")
print(f"Benignos classificados corretamente:          {vp}")
# Imprime o total de erros, destacando o mais grave: um tumor maligno classificado como benigno
print(f"\nTotal de erros: {fp + fn} de {len(y)} pacientes "
      f"({fp} malignos não detectados)")
print("-" * 50)

# Cria o objeto de exibição gráfica da matriz de confusão com os nomes das classes
display = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=nomes_classes)
# Desenha a matriz usando uma escala de azuis e mostrando os valores inteiros em cada célula
display.plot(cmap='Blues', values_format='d')
# Define o título do gráfico
plt.title('Matriz de Confusão - MLP (5-Fold)')
# Renomeia os eixos para português
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Real')
# Ajusta o layout para que nada fique cortado
plt.tight_layout()
# Exibe a janela com o gráfico
plt.show()