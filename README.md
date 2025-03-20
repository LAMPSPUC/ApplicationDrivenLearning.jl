# Examples

This repository contains examples for using ApplicationDrivenLearning.jl.

## Structure


```sh
root/
├─ examples/
│   ├─ newsvendor_1/
│   ├─ eda/ # análises exploratórias
│   └─ models/ # experimentos de modelagem
├─ src/ # código fonte da solução desenvolvida
│   ├─ data_utils/
│   │   ├─ data_read # funções de leitura de dados sanitizados a partir das fontes utilizadas
│   │   ├─ data_transform # funções de tratamento de dados
│   │   ├─ data_save # funções de salvamento de dados processados
│   │   └─ validators
│   │       ├─ input # estruturas de schema para validação dos dados de input
│   │       └─ output # estruturas de schema para validação dos dados de output
│   ├─ models/ # modelos preditivos modularizados
│   ├─ config/ # arquivos YAML de configuração de variáveis gerais do projeto
│   ├─ common/ # clients: código de conexão e interação com serviços externos (s3, blob, keyvault, gpt, etc...)
│   └─ utils/ # funcionalidades úteis
├─ orchestration/
│   ├─ commands/ # arquivos de configuração ou descrição dos processos a serem executados (.yaml, .sh, etc)
│   └─ scripts/ # código dos processos a serem executados
├─ app/ # interface para uso da aplicação (FastAPI, Flask, Streamlit)
├─ tests/ # testes unitários
├─ cicd/ # pipelines de ci/cd
├─ requirements.txt # arquivo de declaração das dependências e versões utilizadas
├─ docs/ # documentação adicional: dicionários de dados, pequenas amostras de dados, instruções de uso, etc
└─ .gitignore
``` 

## Examples Description

### Newvendor 1

Simple multistep newsvendor problem with AR-1 process timeseries. Applies least-squares methodology and BilevelMode and shows difference between ls and opt in in-sample prediction, prediction error and assessed cost. 

### Nesvendor 2

Uses same basic nesvendor problem, but with 2 timeseries representing 2 different newsvendor instances, with different cost parameters and AR-7 processes for timeseries generation. This shows how to use `input_output_map` to apply the same predictive model for multiple prediction decision variables. 

We also analyze the relationship between size of the bias introduced by the application driven learning model, measured by the absolute difference between predictions, and uncertainty from the least-squares model, measured using 95% confidence intervals.

### Nesvendor 3

Uses same problem from `Newsvendor 2`. In this setting, we compare performance for increasing predictive model parameter sizes, showing that GradientMode eventually becomes a better alternative than NelderMeadMode and BilevelMode for big models. 
