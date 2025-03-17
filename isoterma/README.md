
## **Metodología - Detección de Isoterma con MRR**

## **Descripción**
Este script procesa datos del radar **MRR** para detectar y suavizar la isoterma de **0°C** a partir de reflectividad y velocidad de caída.

## **Requisitos**
Necesitas **Python** y las siguientes librerías:
 
```bash
pip install numpy pandas xarray matplotlib proplot statsmodels
```
Si usas `micromamba`:
```bash
micromamba install numpy pandas xarray matplotlib proplot statsmodels
```

## **Uso**
### **Configuración**
Antes de correr el script, asegúrate de ingresar la **ruta correcta** del archivo NetCDF en el código.

### **Ejecución**
Corre el script con:
```bash
python isoterma_rloes_mm_dinámico_combinado.py
```

### **Entradas**
- Archivo **NetCDF** con datos de reflectividad y velocidad de caída.

### **Salidas**
- Gráficos de **reflectividad**, **velocidad** y **gradientes**.
- Isotermas suavizadas con **media móvil** y **LOESS**.

## **Ajustes Manuales**
El código tiene "perillas" ajustables:
- **Reflectividad mínima**: Filtra ruido.
- **Ventana de suavizado**: Controla el grado de suavización.
- **Rango de búsqueda**: Define los límites de altura.
- **Parámetros LOESS**: Ajustan el suavizado final.

Los valores actuales han funcionado bien, pero pueden no ser los mejores para todos los casos. **Prueba diferentes configuraciones** según los datos.

## **Estructura del Código**
1. **Funciones Auxiliares**: Suavizado, filtrado y cálculo de gradientes.
2. **Funciones de Graficado**: Visualización de datos.
3. **Lectura de Datos**: Carga de NetCDF.
4. **Procesamiento**: Cálculo de la isoterma.
5. **Visualización Final**: Ploteo combinado de resultados.

## **Contribuir**
Si quieres mejorar el código, **haz un fork** y envía un **pull request**.

## **Contacto**
Para dudas, **abre un issue** en el repositorio.
```

