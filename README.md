<p align="center">
  <img src="./assets/img/jurai-git.png"/>
</p>

---

<h1 align="center">JurAI - Models</h1> <h2 align="center">Repositório oficial dos modelos de redes neurais do JurAI.</h2>

<div align="center">
    
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-00000F?style=for-the-badge&logo=mysql&logoColor=white)
    
</div>

## Descrição

O JurAI é um projeto dedicado ao desenvolvimento de modelos de redes neurais voltados para a área jurídica. 
Este repositório contém todos os arquivos necessários para o treinamento dos nossos modelos de IA, 
desconsiderando os conjuntos de dados, com foco na aplicação em textos jurídicos e análise de argumentos legais.

## Estrutura do Repositório

    tools/ - Ferramentas para gerenciamento de conjuntos de dados.
    tools/eda - Classes para análise exploratória de dados.
    models/ - Arquivos e pastas dos modelos de redes neurais.
    datasets/ - Pasta para conjuntos de dados e logs de treinamento.

## Instalação

Para começar a utilizar o projeto do JurAI, clone este repositório e instale as dependências necessárias.

~~~bash
git clone https://github.com/jurai-git/jurai-models.git
cd jurai-models
pip install -r requirements.txt
~~~

## Configuração:
### Bash
1. Abra o projeto no terminal Linux, ou no git bash para Windows.
2. Execute `chmod +x ./configure.sh` 
3. Execute `./configure.sh <db_host> <db_user> <db_password> <db_name> <db_table>`

### Manualmente
1. Na raíz do projeto, crie um arquivo dotenv (.env) para as credenciais do banco de dados e o caminho da pasta de conjuntos de dados, assim como:
    ~~~env
    DB_HOST=localhost
    DB_USER=root
    DB_PASSWORD=root@123
    DB_NAME=jurai
    DB_TABLE=processos

    PROJECT_PATH=~/JurAI/jurai-models
    DATASET_PATH=~/JurAI/jurai-models/datasets
    ~~~

## Tecnologias:
* TensorFlow
* Keras
* PyTorch

## Licença

Este projeto é licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Contato

Para mais informações, entre em contato através do email: contas.jurai@gmail.com
