# 💡 Red Neuronal con NumPy 🧩

**Estudiante**: Aarón Enrique Ramírez González  

**Tarea 3**: Red Neuronal con Numpy

**Materia**: Sistemas de Visión Artificial  


---

## 📝 Descripción

Este repositorio contiene la implementación de una **red neuronal desde cero** utilizando **NumPy** para clasificar datos generados a partir de distribuciones gaussianas. El proyecto incluye:

- 🧩 **Generación de datos**: Creación de un conjunto de datos sintético para clasificación binaria.
- 🛠️ **Construcción de la red neuronal**: Implementación de una red con múltiples capas y funciones de activación.
- 🚀 **Entrenamiento y evaluación**: Entrenamiento de la red neuronal y visualización de los resultados.

El código está comentado paso a paso para una mayor comprensión.

---

## 📋 Requisitos

Para ejecutar este proyecto, necesitas tener instaladas lo siguiente:
- **Python**: En este caso la versión utilizada para la realización de esta tarea es la 3.12.4
- **NumPy**: Para cálculos numéricos y manejo de arreglos.
- **Matplotlib**: Para la generación de gráficas.
- **Scikit-learn**: Para generar datos sintéticos.

Puedes instalar estas dependencias utilizando `pip`:

```bash
pip install numpy matplotlib scikit-learn
```
## 🗂️ Estructura del Proyecto
El proyecto está organizado de la siguiente manera:

``` bash
TAREA_3/
│
├── src/
│   └── Neuronal_Network_Numpy.py  # Script principal de la red neuronal
│
├── .gitignore      # Archivo para ignorar archivos no deseados
├── main.py         # Script principal para ejecutar el proyecto
├── README.md       # Este archivo
└── Requirements.txt # Lista de dependencias del proyecto
```
## 🚀 ¿Cómo usar este repositorio?
Sigue estos pasos para ejecutar el proyecto en tu lab:

### Clona el repositorio 🖥️:
Abre una terminal y ejecuta el siguiente comando para clonar el repositorio en tu computadora:

```bash
git clone https://github.com/Aaron040-Rmz/Neuronal_Network_Numpy_2230294
```
### Cree un nuevo entorno virtual
Se recomienda tener el entorno virtual generado en la carpeta principal para un fácil acceso, su activación y desactivación se realiza de la siguiente forma:

En PowerShell:
```
.\nombre_del_entorno\Scripts\Activate
deactivate
```
En Unix:
```
source nombre_del_entorno/bin/activate
deactivate
```
### Instala las dependencias 📦:
Asegúrate de tener instaladas las bibliotecas necesarias. Ejecuta el siguiente comando para instalarlas:

```bash
pip install -r Requirements.txt
```
### Ejecuta el script principal🚀:
Para entrenar y evaluar la red neuronal, ejecuta:

```bash
python main.py
```
### Visualiza los resultados 📊:

  * El script mostrará el error durante el entrenamiento en la consola.

  * También se mostrará un gráfico con los datos originales y los datos clasificados por la red neuronal. 👀

## 🛠️ Tecnologías Utilizadas
**Python**: Lenguaje de programación principal en este caso se utilizó la versión 3.12.4.

**NumPy**: Para cálculos numéricos y manejo de arreglos.

**Matplotlib**: Para visualización de datos y gráficos.

**Scikit-learn**: Para generar datos sintéticos.

## 🧑‍💻 ¿Qué hace el código?
El código realiza lo siguiente:

1. **Genera un conjunto de datos sintético** utilizando distribuciones gaussianas.
 
2. **Define funciones de activación**:

    * **Sigmoid**: Para la capa de salida.

    * **ReLU**: Para las capas ocultas.

3. **Inicializa los parámetros** de la red neuronal (pesos y sesgos).

4. **Entrena la red neuronal** utilizando propagación hacia adelante y hacia atrás.

5. **Evalúa el modelo** clasificando nuevos datos y visualizando los resultados.

## 🙏 Agradecimientos
¡Gracias por leer mi "readme"!😃

Como lo hice en los anteriores repositorios, te deseo un muy buen día y que Dios te bendiga en gran manera a ti y a toda tu familia 😊
