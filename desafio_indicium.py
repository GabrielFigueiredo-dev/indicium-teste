import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from scipy.stats import f_oneway

# Carregamento do dataset
data = pd.read_csv('teste_indicium_precificacao.csv', encoding='utf-8', on_bad_lines='skip')

# Verificação do número de colunas existentes em cada linha
num_colunas_esperado = 16
with open('teste_indicium_precificacao.csv', 'r', encoding='utf-8') as f:
    for i, linha in enumerate(f):
        num_colunas = len(linha.split(','))
        if num_colunas != num_colunas_esperado:
            print(f"Linha {i+1} tem {num_colunas} colunas")

# Garantia de que todas as linhas tenham o número correto de colunas
data = data[data.apply(lambda x: len(x) == num_colunas_esperado, axis=1)]

# Informações gerais do dataset
print(data.head())
print(data.info())

# Contagem de valores ausentes em cada coluna
missing_values = data.isnull().sum()

# Exibição de colunas que possuem valores ausentes
missing_values = missing_values[missing_values > 0]
print("\nValores Ausentes no Dataset:")
print(missing_values)

# Preencher valores ausentes
data.loc[:, 'nome'] = data['nome'].fillna('Sem Nome')
data.loc[:, 'host_name'] = data['host_name'].fillna('Desconhecido')
data.loc[:, 'reviews_por_mes'] = data['reviews_por_mes'].fillna(0)
data['ultima_review'] = pd.to_datetime(data['ultima_review'])

# Estatísticas descritivas para verificação de valores discrepantes
print("\nEstatísticas Descritivas:")
print(data[['price', 'minimo_noites']].describe())

# Boxplot do preço para visualização de outliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=data['price'])
plt.title("Boxplot de Preços")
plt.xlabel("Preço ($)")
plt.savefig("boxplot_precos.png")
plt.show()

# Boxplot do número mínimo de noites
plt.figure(figsize=(8, 5))
sns.boxplot(x=data['minimo_noites'])
plt.title("Boxplot do Número Mínimo de Noites")
plt.xlabel("Mínimo de Noites")
plt.savefig("boxplot_minimo_noites.png")
plt.show()

# Remoção de outliers
outliers_price = 1000
data = data[data['price'] <= outliers_price]
data = data[data['minimo_noites'] <= 365]

# Pós tratamento
# Estatísticas descritivas
print("Estatísticas Descritivas Após Limpeza:")
print(data.describe())

# Gráficos Exploratórios
# Gráfico 1 - Distribuição de preços
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], bins=50, kde=True, color='blue')
plt.title("Distribuição de Preços", fontsize=16)
plt.xlabel("Preço ($)")
plt.ylabel("Frequência")
plt.savefig("distr_precos.png")
plt.show()

# Gráfico 2 - Contagem de anúncios por tipo de quarto
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='room_type', palette='viridis', hue='room_type', dodge=False)
plt.title("Contagem de Anúncios por Tipo de Quarto", fontsize=16)
plt.xlabel("Tipo de Quarto")
plt.ylabel("Contagem")
plt.legend([], [], frameon=False)
plt.savefig("anuncios_por_tipo.png")
plt.show()

# Gráfico 3 - Boxplot de preços por bairro_group
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x='bairro_group', y='price', hue='bairro_group', palette='coolwarm', legend=False)
plt.title("Dispersão de Preços por Bairro Group", fontsize=16)
plt.xlabel("Bairro Group")
plt.ylabel("Preço ($)")
plt.savefig("precos_por_bairro.png")
plt.show()

# Gráfico 4 - Mapa de preços geográfico
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='longitude', y='latitude', hue='price', palette='coolwarm', alpha=0.6)
plt.title("Distribuição Geográfica de Preços", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title='Preço ($)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("distr_grafica_precos.png")
plt.show()

# WordCloud para locais de alto valor
high_price = data[data['price'] > data['price'].quantile(0.90)]
text = " ".join(high_price['nome'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Palavras mais comuns em nomes de locais de alto valor", fontsize=16)
plt.savefig("glossario.png")
plt.show()

# Seleção das variáveis explicativas e variáveis alvo
X = data[['bairro_group', 'room_type', 'latitude', 'longitude', 'minimo_noites',
          'numero_de_reviews', 'reviews_por_mes', 'calculado_host_listings_count',
          'disponibilidade_365']]
y = data['price']

# Transformação de variáveis categóricas (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# Divisão dos dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regressão Linear
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)
mae_lin = mean_absolute_error(y_test, y_pred_lin)
rmse_lin = mean_squared_error(y_test, y_pred_lin) ** 0.5  
print(f"Regressão Linear - MAE: {mae_lin:.2f}, RMSE: {rmse_lin:.2f}")

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5  
print(f"Random Forest - MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}")

# Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = mean_squared_error(y_test, y_pred_gb) ** 0.5  
print(f"Gradient Boosting - MAE: {mae_gb:.2f}, RMSE: {rmse_gb:.2f}")

# Melhor modelo a ser utilizado, ou seja, menor RMSE
if rmse_rf < rmse_gb and rmse_rf < rmse_lin:
    best_model = rf_model
    model_name = 'Random Forest'
elif rmse_gb < rmse_rf and rmse_gb < rmse_lin:
    best_model = gb_model
    model_name = 'Gradient Boosting'
else:
    best_model = lin_model
    model_name = 'Regressão Linear'

joblib.dump(best_model, 'melhor_modelo_precificacao.pkl')
print(f"Modelo escolhido: {model_name}. Salvo como 'melhor_modelo_precificacao.pkl'")

# Matriz de correlação
numeric_data = data.select_dtypes(include=[float, int])
correlation_matrix = numeric_data.corr()
print("Matriz de Correlação:")
print(correlation_matrix['price'].sort_values(ascending=False))

# Preços médios por bairro
mean_prices = data.groupby('bairro_group')['price'].mean()
print(f"Preços médios por bairro: {mean_prices}")

# Correlações específicas
corr_min_nights = data['minimo_noites'].corr(data['price'])
corr_availability = data['disponibilidade_365'].corr(data['price'])
print(f"Correlação entre mínimo de noites e preço: {corr_min_nights}")
print(f"Correlação entre disponibilidade e preço: {corr_availability}")

# Palavras comuns em locais de alto valor
common_words = " ".join(high_price['nome'].dropna())
print("Palavras comuns em locais de alto valor:")
print(common_words)

# Previsão de preço para novos apartamentos
novo_apartamento = pd.DataFrame([{
    'bairro_group': 'Manhattan',
    'room_type': 'Entire home/apt',
    'latitude': 40.75362,
    'longitude': -73.98377,
    'minimo_noites': 1,
    'numero_de_reviews': 45,
    'reviews_por_mes': 0.38,
    'calculado_host_listings_count': 2,
    'disponibilidade_365': 355
}])

# Transformação de variáveis categóricas em dummies
novo_apartamento_dummies = pd.get_dummies(novo_apartamento, drop_first=True)

# Garantia de que todas as colunas necessárias estejam presentes
for col in X_train.columns:
    if col not in novo_apartamento_dummies.columns:
        novo_apartamento_dummies[col] = 0

# Reorganização de colunas para corresponder ao treinamento
novo_apartamento_dummies = novo_apartamento_dummies[X_train.columns]

price_pred = rf_model.predict(novo_apartamento_dummies)
print(f"Preço sugerido: ${price_pred[0]:.2f}")