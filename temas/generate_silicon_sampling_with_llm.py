import os
from io import StringIO

import pandas as pd
from google import genai
from google.genai import types
from sklearn.utils import shuffle


def generate_content(
    prompt: str,
    model: str = "gemini-2.5-flash-lite",
    system_instruction: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    genai_api_key = os.getenv("GOOGLE_API_KEY")
    if not genai_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")

    with genai.Client(api_key=genai_api_key) as client:
        response = client.models.generate_content(
            model=model,
            contents=types.Part.from_text(text=prompt),
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                top_p=top_p,
            ),
        )
        return response.text


few_shot_prompt = """
Prepare um conjunto de dados estruturados no formato `.csv` no idioma português (pt_BR) com as dimensões:
- **index**: Valor numérico sequêncial iniciando em 1;
- **review_text**: Um texto entre 5 e 25 palavras contendo a opinião sobre um filme qualquer;
- **polarity**: A intenção do comentário, onde 0 representa negativo e 1 representa positivo.

Cada linha deve simular a avaliação de uma pessoa diferente, com suas características únicas e pessoais.
O público respondente possui faixa etária de 20 à 45 anos, e variação demográfica (genêro, contexto social, crença, etc).

**Exemplos**

Usuário: Prepare um conjunto de 5 analises positivas e 5 análises negativas
Resposta:
```csv
{first_example}
```

Usuário: Prepare um conjunto de 3 analises positivas e 5 análises negativas
Resposta:
```csv
{second_example}
```

**Solicitação**
Usuário: Prepare um conjunto de {n_positive} análises positivas e {n_negative} análises negativas
Resposta:
""".strip()


def simulate_review(
    n_positive: int, n_negative: int, first_example: str, second_example: str
) -> pd.DataFrame:
    prompt = few_shot_prompt.format(
        first_example=first_example,
        second_example=second_example,
        n_positive=n_positive,
        n_negative=n_negative,
    )
    response = generate_content(
        prompt=prompt, model="gemini-2.5-flash-lite", temperature=0.3, top_p=1.0
    )

    if response.startswith("```csv"):
        response = response[6:]

    if response.endswith("```"):
        response = response[:-3]

    buffer = StringIO(response.strip())
    review_df = pd.read_csv(buffer)
    review_df = shuffle(review_df, random_state=42).reset_index(drop=True)
    return review_df[["review_text", "polarity"]]


first_example = """index,review_text,polarity
1,"que filme sensível e encantador. paisagens, personagens, trilha, tudo muito gostoso de assistir. ps: só o Oliver que é meio chato mesmo.",1
2,"o melhor filme nacional que assisti nos últimos 3 anos. sem sombra nenhuma de dúvidas. a trilha sonora? nem se fale...",1
3,"melhor filme da trilogia ,sem sombra de duvidas",1
4,"Um belo filme.",1
5,"Fui ver sem esperar muito, ainda mais depois de não curtir muito alguns filmes do Denis e por achar que alguns filmes dele possuem algumas pontas bem soltas. Esse é diferente: é totalmente cíclico em todos os sentidos. Redondo, fechado. Muito bom!",1
6,"Devo confessar a minha imensa decepção com esse filme que eu consideraria extremamente superestimado. O roteiro é besta, arrastado, a trilha sonora é enjoativa, alguns personagens dispensáveis, final sem graça demais para quem passou quase 2 hrs esperando por algo surpreendente, etc. A única coisa que salva-se no filme, na minha opinião, é a atuação do DeNiro. Mas como já foi dito anteriormente, atuação não salva filme. A verdade é que o filme é chato, cansativo, tão chato quanto estudar. Se passasse uma mosca eu iria perder a concentração. Mais um filme modinha! Decepcionante!",0
7,"Filme morno. Mesmo com o elenco que tem o filme não tem empatia. Filme perca de tempo total.",0
8,"Enredo muito sem graça e clichês piores ainda. Esperava mais...",0
9,"Não gostei nem um pouco.",0
10,"Onde fica esse lugar incrível onde foi filmada a rave e tal? Eu me apaixonei pela locação <3. Odiei o fato de colocarem que todos que frequentassem esse tipo de festa/balada/rave como usuários de drogas, generalizou bastante. A fotografia é bem bonita, mas o filme poderia ser melhor desenvolvido... Enfim. Maomenos.",0"""

second_example = """index,review_text,polarity
1,"Uma palavra pra essa animação : indescritível. Filme que simplesmente me deixou sem palavras de tão bem feito.",1
2,"Adorei, adorei, adorei. :}",1
3,"Bem divertido! Inesperado, meio triste, reflexivo... e morre o homem branco com toda a prata que ajuntou, diria Tonto. E não sei porque teimam em igualar Tonto com Jack Sparrow - a personalidade dos dois é completamente diferente: melancólica e sábia versus irreverente e desonesto. Vale também pelas cenas absurdas que nem vou comentar, mas são hilárias kkkkkkk",1
4,"Que engraçado o jeito de falar...nao se entende nada....",1
5,"É estranho o desenrolar das coisas no fim do documentário ele consegue ligar o cara a Jane através de um site que ele tinha em mãos o tempo todo, não apenas isso como o gordão se entregou em uma bandeja; me senti num filme que perdeu tempo demais no desenvolvimento e precisou acabar rápido.",1
6,"Muito estranho, não sei bem como explicar, mas a trama se desenrola muito ""nas coxas"". Melhor parte do filme inteiro é o Willem como Bobby Peru. É um trash com atores de certo porte. Em suma é Insano e sem sentido algum! O que não deixa de ser uma aventura divertida ""só pra descontrair"".",0
7,"O filme começa bem e vai piorando. Entendo pq muitas pessoas gostaram desse filme, foi exatamente o mesmo motivo de eu não ter gostado. Um filme feito pra crítica, bastante sangue pra imitar o Tarantino, música lerda (ou silêncio) na hora da ação, pra fingir que a violência é arte.",0
8,"Uma das piores coisas que assisti ultimamente. Eu realmente não tenho nada para comentar.",0
9,"Vale pela cena da moto e do helicóptero, o resto a gente guarda na fanbase",0
10,"é tanta futilidade que custei a terminar. a Rachel é reduzida a mais uma mulherzinha que tem que se encaixar pra agradar a família nojenta do macho, sendo uma mulher inteligente e realizada profissionalmente. o Nick é um egoísta covarde que não teve coragem de dizer, para a mulher que ele diz que ""ama"", o que é a família dele é de verdade. desperdício de elenco. Peik Lin melhor personagem e amei a voz roca da atriz.",0"""


if __name__ == "__main__":
    review_df = simulate_review(
        n_positive=100,
        n_negative=100,
        first_example=first_example,
        second_example=second_example,
    )

    # Salvar o DataFrame em um arquivo CSV
    review_df.to_csv("simulated_reviews.csv", index=False)
