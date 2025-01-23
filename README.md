# Sistema de Recomendação de Receitas

Este repositório contém um protótipo de um sistema de recomendação para um site de receitas fictício. O objetivo é recomendar receitas com base nas preferências do usuário, utilizando **Sentence Transformers** para representação semântica e um modelo de mistura gaussiana para agrupamento de similaridade.

## Funcionalidades

1. **Base de Dados:**
   - A base de dados é representada por um arquivo JSON contendo informações sobre receitas, incluindo títulos, descrições e tags associadas.
   - Um preview da estrutura do arquivo:
     ```json
     [
       {
      		"directions": [
      			"1. Place the stock, lentils, celery, carrot, thyme, and salt in a medium saucepan and bring to a boil. Reduce heat to low and simmer until the lentils are tender, about 30 minutes, depending on the lentils. (If they begin to dry out, add water as needed.) Remove and discard the thyme. Drain and transfer the mixture to a bowl; let cool.",
      			"2. Fold in the tomato, apple, lemon juice, and olive oil. Season with the pepper.",
      			"3. To assemble a wrap, place 1 lavash sheet on a clean work surface. Spread some of the lentil mixture on the end nearest you, leaving a 1-inch border. Top with several slices of turkey, then some of the lettuce. Roll up the lavash, slice crosswise, and serve. If using tortillas, spread the lentils in the center, top with the turkey and lettuce, and fold up the bottom, left side, and right side before rolling away from you."
		],
		"fat": 7.0,
		"date": "2006-09-01T04:00:00.000Z",
		"categories": [
			"Sandwich",
			"Bean",
			"Fruit",
			"Tomato",
			"turkey",
			"Vegetable",
			"Kid-Friendly",
			"Apple",
			"Lentil",
			"Lettuce",
			"Cookie"
		],
		"calories": 426.0,
		"desc": null,
		"protein": 30.0,
		"rating": 2.5,
		"title": "Lentil, Apple, and Turkey Wrap ",
		"ingredients": [
			"4 cups low-sodium vegetable or chicken stock",
			"1 cup dried brown lentils",
			"1/2 cup dried French green lentils",
			"2 stalks celery, chopped",
			"1 large carrot, peeled and chopped",
			"1 sprig fresh thyme",
			"1 teaspoon kosher salt",
			"1 medium tomato, cored, seeded, and diced",
			"1 small Fuji apple, cored and diced",
			"1 tablespoon freshly squeezed lemon juice",
			"2 teaspoons extra-virgin olive oil",
			"Freshly ground black pepper to taste",
			"3 sheets whole-wheat lavash, cut in half crosswise, or 6 (12-inch) flour tortillas",
			"3/4 pound turkey breast, thinly sliced",
			"1/2 head Bibb lettuce"
		],
		"sodium": 559.0}
       ]
     ```

2. **Geração de Embeddings:**
   - As tags de cada receita são convertidas em embeddings vetoriais usando o **Sentence Transformers**.

3. **Sistema de Recomendação:**
   - Usuários simulam preferências ao selecionar tags favoritas.
   - O sistema compara os embeddings das tags favoritas com os das receitas e retorna as **Top-K** receitas mais relevantes.

4. **Clustering de Similaridade:**
   - Um modelo de mistura gaussiana é usado para identificar clusters de similaridade entre receitas, permitindo recomendações mais precisas.

## Breve Explicação sobre Mistura Gaussiana

A mistura gaussiana é um modelo probabilístico que assume que os dados são distribuídos em múltiplos clusters, onde cada cluster segue uma distribuição gaussiana. Este método é usado para:

- Modelar a similaridade entre embeddings.
- Estimar as probabilidades de uma receita pertencer a um cluster específico com base em suas tags.
- Melhorar a eficiência e a relevância das recomendações.

## Dependências

Certifique-se de ter as seguintes dependências instaladas:

- **torch:** Para utilização do Sentence Transformers.
- **pandas:** Manipulação e análise de dados estruturados.
- **numpy:** Operações matemáticas e vetoriais.

Instale todas as dependências executando:
```bash
pip install torch pandas numpy
```

## Como Usar

1. Clone o repositório:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DO_REPOSITORIO>
   ```

2. Certifique-se de que o arquivo `receitas.json` está no diretório raiz, contendo a base de dados no formato descrito acima.

3. Execute o script principal para gerar recomendações:
   ```bash
   python execute.py
   ```

4. Insira as tags favoritas simulando as preferências do usuário. O sistema retornará as Top-K receitas recomendadas.

## Estrutura do Repositório

```
.
├── receitas.json       # Base de dados de receitas
├── execute.py          # Script de somulação
├── recommend_me.py     # Script Principal
└── README.md          # Documentação
```

## Contribuições

Sinta-se à vontade para abrir issues ou enviar pull requests com melhorias, novas funcionalidades ou correções.

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.

