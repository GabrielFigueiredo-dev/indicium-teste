# Projeto de Precificação com Machine Learning

Repositório com o código da Prova Técnica do Programa LIGHTHOUSE da Indicium.

Este projeto realiza a análise e previsão de preços utilizando um conjunto de dados sobre hospedagens.

## Conteúdos

:small_blue_diamond:  [Estrutura do projeto](#estrutura-do-projeto)
:small_blue_diamond:  [Requisitos](#requisitos)
:small_blue_diamond:  [Instalação](#instalação)
:small_blue_diamond:  [Como usar](#como-usar-arrow_forward)
:small_blue_diamond:  [Exercício Técnico](#exercício-técnico-pencil)
:small_blue_diamond:  [Tecnologias](#tecnologias-books)
:small_blue_diamond:  [Autor](#autor)

## Estrutura do projeto 
- `desafio_indicium.py`: código de execução de ETL dos dados além das análises;
- `LH_CD_Gabriel Vinícius de Figueiredo.pdf`: relatório de análise exploratória de dados e respostas referentes ao desafio;
- `requirements.txt`: requisitos para execução do projeto;
- `teste_indicium_precificacao.csv`: conjunto de dados sobre hospedagens;

## Requisitos

Certifique-se de ter instalado os seguintes pacotes Python, mas não se preocupe, faremos o passo a passo para a utilização do projeto:

- contourpy==1.3.1
- cycler==0.12.1
- fonttools==4.55.8
- joblib==1.4.2
- kiwisolver==1.4.8
- matplotlib==3.10.0
- numpy==2.2.2
- packaging==24.2
- pandas==2.2.3
- pillow==11.1.0
- pyparsing==3.2.1
- python-dateutil==2.9.0.post0
- pytz==2024.2
- scikit-learn==1.6.1
- scipy==1.15.1
- seaborn==0.13.2
- six==1.17.0
- threadpoolctl==3.5.0
- tzdata==2025.1
- wordcloud==1.9.4

## Instalação

    ```sh
     # Clone o repositório:
     git clone https://github.com/gab-figueiredo/indicium-teste.git
     
     # Entre no diretório do repositório:
     cd indicium-teste/
    
    ```

## Como usar :arrow_forward:

    ```sh
      # Instala, cria e ativa virtualenv:
      pip install virtualenv
      virtualenv venv
      . venv/bin/activate

      # Instalando requirements
      pip install -r requirements.txt
    
      # Rodando exercício:
      python desafio_indicium.py
      
    ```
## Exercício Técnico :pencil:

Durante a execução do código inúmeras imagens de análise serão mostradas na tela ou adicionadas a pasta atual, por exemplo:

- Análise exploratória com:
  a. gráficos de box_plot;
  b. gráficos de distribuição;
  c. glossários;

O exercício também conta com:
  1. Construção e Avaliação dos Modelos:
     - Regressão Linear:
       ```sh
          lin_model = LinearRegression()
          lin_model.fit(X_train, y_train)
       ```
     - Random Forest:
       ```sh
          rf_model = RandomForestRegressor(random_state=42)
          rf_model.fit(X_train, y_train)
       ```
     - Gradient Boosting:
       ```sh
          gb_model = GradientBoostingRegressor(random_state=42)
          gb_model.fit(X_train, y_train)
        ```
  2. Geração de Gráficos e WordCloud:

     ```sh
        high_price = data[data['price'] > data['price'].quantile(0.90)]
        text = " ".join(high_price['nome'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig("glossario.png")
        plt.show()
      ```
  3. Previsão para Novos Apartamentos:
      ```sh
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
  
        price_pred = rf_model.predict(novo_apartamento_dummies)
        print(f"Preço sugerido: ${price_pred[0]:.2f}")
      
      ```
 Ao final da execução, o modelo desenvolvido será salvo no arquivo `melhor_modelo_precificacao.pkl`;
  

## Tecnologias :books:
As seguintes ferramentas foram usadas na construção do projeto:

- [Python](https://www.python.org/downloads/release/python-3120/): Linguagem principal usada para o projeto.
- [Pandas](https://pandas.pydata.org/docs/dev/whatsnew/v2.2.0.html): Biblioteca para manipulação e análise de dados.
- [Matplotlib](https://matplotlib.org/stable/index.html) e [Seaborn](https://seaborn.pydata.org/installing.html): Bibliotecas para criação de gráficos.
- [WordCloud](https://pypi.org/project/wordcloud/): Biblioteca para manipulação e análise de dados.
- [Scikit-learn](https://scikit-learn.org/dev/whats_new/v1.6.html): Biblioteca para construção e avaliação de modelos de aprendizado de máquina.
- [Joblib](https://joblib.readthedocs.io/en/stable/installing.html): Biblioteca para salvar e carregar modelos treinados.

## Autor

👤 **Gabriel Figueiredo**

[![Linkedin Badge](https://img.shields.io/badge/-Gabriel-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/gabrielvinifigueiredo/)](https://www.linkedin.com/in/gabrielvinifigueiredo/) [![Gmail Badge](https://img.shields.io/badge/-gabrielfigueiredo158@gmail.com-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:gabrielfigueiredo158@gmail.com)](mailto:gabrielfigueiredo158@gmail.com)
