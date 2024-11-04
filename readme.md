Para correr la pipeline se necesita antes de correr el experimento tener el archivo parmas.yaml de la siguiente estructura, el proceso añade secciones dinamicas al archivo por lo cual al momento de volver a correrlo podria romperse la pipeline:

### Version inicial del params.yaml
```yaml
train:
  TEST_SIZE: 0.3
  VALIDATE_SIZE: 0.2
  RANDOM_STATE: 2024
  CV: 5
  alpha: 0.1
preprocessing:
  target: price
  features:
  - area
  - bedrooms
  - bathrooms
  - stories
  - mainroad
  - guestroom
  - basement
  - hotwaterheating
  - airconditioning
  - parking
  - prefarea
  - furnishingstatus
  ```

### Comandos
dvc repro -f