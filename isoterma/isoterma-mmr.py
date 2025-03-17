import xarray as xr
from matplotlib import colors
import matplotlib.pyplot as plt
import proplot as pplt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

# ----------------------------------------------------------------------------
# 1) FUNCIONES AUXILIARES GENERALES
# ----------------------------------------------------------------------------

def suavizado_media_movil(datos, ventana=5):
    """
    Aplica un filtro de media móvil a los datos.
    """
    return np.convolve(datos, np.ones(ventana)/ventana, mode='same')

def filtrar_isoterma_loess(y, x=None, frac=0.05, it=3):
    """
    Aplica un filtrado robusto LOESS (RLOESS) a la serie de datos 'y'.
    
    Parámetros:
    - y (array-like): Datos a filtrar.
    - x (array-like, opcional): Eje de la variable independiente. 
      Si no se provee, usa np.arange(len(y)).
    - frac (float): Porcentaje de la ventana para LOESS. 
      Más alto => más suavizado.
    - it (int): Número de iteraciones robustas.

    Retorna:
    - numpy.ndarray: Datos suavizados con LOESS robusto.
    """
    if x is None:
        x = np.arange(len(y))
    resultado_loess = lowess(y, x, frac=frac, it=it, return_sorted=False)
    return resultado_loess

def add_no_data(ax, times, xlim):
    """
    Dibuja un área sombreada en caso de no datos.
    """
    cs = ax.contourf([times[0], times[-1]], [0,600], np.ones((2,2)), 
                     colors='none', edgecolor='red', hatches=['////'], zorder=0)
    for coll in cs.collections:
        coll.set_edgecolor('gray3')
    ax.text(xlim[0], 300, 'Sin Datos',
            color='gray8',
            ha='left', va='center',
            zorder=1)

def calcular_gradiente(datos, marco):
    """
    Calcula el gradiente vertical aproximado con una pequeña ventana 'marco'.
    """
    pesos = np.array([marco - i for i in range(marco)], dtype=np.float64)
    pesos = pesos / np.sum(pesos)
    niveles_restantes = datos.shape[0] - 1 - 2*(marco - 1)
    gradiente_datos = np.zeros((niveles_restantes, datos.shape[1]))

    for i in range(marco, datos.shape[0] - marco):
        superior = np.sum([pesos[j]*datos[i + j, :] for j in range(marco)], axis=0)
        inferior = np.sum([pesos[j]*datos[i - j - 1, :] for j in range(marco)], axis=0)
        gradiente_datos[i - marco, :] = superior - inferior

    filas_superior = marco
    filas_inferior = marco - 1
    gradiente_datos_completo = np.pad(gradiente_datos,
                                      ((filas_superior, filas_inferior), (0, 0)),
                                      mode='constant',
                                      constant_values=0)
    return gradiente_datos_completo

def detectar_isoterma_cero_acotada(gradiente, heights_ajustado, hmin, hmax):
    """
    Busca el mínimo del gradiente en cada instante de tiempo,
    pero solo dentro de [hmin, hmax].
    """
    i_min = np.searchsorted(heights_ajustado, hmin, side='left')
    i_max = np.searchsorted(heights_ajustado, hmax, side='right') - 1
    i_max = min(i_max, len(heights_ajustado) - 1)

    resultados = []
    for t in range(gradiente.shape[1]):
        sub_grad = gradiente[i_min : i_max+1, t]
        if np.all(sub_grad == sub_grad[0]):
            # Si no hay variación, no se detecta isoterma válida
            resultados.append({
                "iter": t,
                "altura_minima": np.nan,       # Usamos NaN para indicar ausencia
                "gradiente_minimo": np.nan
            })
        else:
            idx_sub_min = np.argmin(sub_grad)
            idx_global_min = i_min + idx_sub_min
            grad_min = gradiente[idx_global_min, t]
            alt_min = heights_ajustado[idx_global_min]
            resultados.append({
                "iter": t,
                "altura_minima": alt_min,       # guardamos la altura real
                "gradiente_minimo": grad_min
            })
    return resultados

# ----------------------------------------------------------------------------
# 2) FUNCIONES DE GRAFICADO GENERALES
# ----------------------------------------------------------------------------

def crear_colormap(custom_colors, set_bad_color='0.9', bounds=np.arange(-5,50,1)):
    """
    Crea un colormap personalizado.
    """
    cmap = colors.LinearSegmentedColormap.from_list("custom", custom_colors)
    cmap.set_bad(set_bad_color, 1)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

def plot_mrr(xlim, times, heights, data, tipo='reflectividad', isoterma=None, hora_local=False, colorhex=None):
    """
    Función genérica para graficar reflectividad o velocidad de caída.

    Parámetros:
    - tipo (str): 'reflectividad' o 'velocidad'.
    - data (dict): Diccionario con claves 'main' y opcionalmente 'additional'.
    - isoterma (array-like, opcional): Datos de la isoterma para superponer.
    """
    if colorhex is None:
        if tipo == 'reflectividad':
            colorhex = ['#ccd8ff','#3366ff','#9fdf9f','#00b300','#ffff00','#ffcc30',
                        '#e62e00','#ff6600','#fff0e5','#c03fc0','#602060']
            bounds = np.arange(-5,50,1)
            label = '[dBZ]'
            title_main = 'Reflectividad Equivalente'
            suptitle = 'Radar Perfilador MRR en UOH Rancagua'
        elif tipo == 'velocidad':
            colorhex = ['#ccd8ff','#3366ff','#9fdf9f','#00b300','#ffff00','#ffcc30',
                        '#e62e00','#ff6600','#fff0e5','#c03fc0','#602060']
            bounds = np.arange(-10,15,1)
            label = '[m/s]'
            title_main = 'Velocidad de Caída'
            suptitle = 'Radar Perfilador MRR en UOH Rancagua'
        else:
            raise ValueError("Tipo no soportado. Use 'reflectividad' o 'velocidad'.")
    else:
        bounds = np.arange(-5,50,1) if tipo == 'reflectividad' else np.arange(-10,15,1)
        label = '[dBZ]' if tipo == 'reflectividad' else '[m/s]'
        title_main = 'Reflectividad Equivalente' if tipo == 'reflectividad' else 'Velocidad de Caída'
        suptitle = 'Radar Perfilador MRR en UOH Rancagua'

    cmap, norm = crear_colormap(colorhex, bounds=bounds)

    xlabel = r'Hora Local $\rightarrow$' if hora_local else r'Hora UTC $\rightarrow$'
    ylim = [0, 3600] if heights[-1] < 5000 else [0, 8000]

    total_seconds = (xlim[1] - xlim[0]).total_seconds()
    if total_seconds <= 14400:
        xlocator, xminorlocator = ('hour', range(0,24,1)), ('minute',30)
    elif 14400 < total_seconds <= 82800.0:
        xlocator, xminorlocator = ('hour', range(0,24,3)), ('hour', range(0,24,1))
    else:
        xlocator, xminorlocator = ('hour', range(0,24,6)), ('hour', range(0,24,2))

    if 'additional' not in data or data['additional'] is None:
        fig, ax = pplt.subplots(refwidth=5, refaspect=3)
        m = ax.pcolormesh(times, heights, data['main'],  
                          norm=norm, cmap=cmap, shading='auto')
        if isoterma is not None:
            ax.plot(times, isoterma, color='black', linestyle='--', linewidth=1.5, label='Isoterma Dinámica')
            ax.legend(loc='upper right', fontsize=10)
        ax.format(ultitle=title_main,
                  xrotation=False,
                  xformatter='concise',
                  xlocator=xlocator,
                  xminorlocator=xminorlocator,
                  ylim=ylim,
                  yticklabelloc='both',
                  ytickloc='both',
                  xticklabelsize=8,
                  suptitle=suptitle,
                  ylabel='Altitud [msnm]',
                  xlabel=xlabel)
        ax.colorbar(m, loc='r', label=label, length=0.7)
        if xlim != '':
            ax.format(xlim=xlim)
    else:
        fig, ax = pplt.subplots(nrows=2, refwidth=5, refaspect=3)
        m_main = ax[0].pcolormesh(times, heights, data['main'],
                                  norm=norm, cmap=cmap, shading='auto')
        add_no_data(ax[0], times, xlim)
        m_add = ax[1].pcolormesh(times, heights, data['additional'],
                                 vmin=-3, vmax=10, cmap='RdBu', shading='auto')
        add_no_data(ax[1], times, xlim)

        if isoterma is not None:
            ax[0].plot(times, isoterma, color='black', linestyle='--', linewidth=1.5, label='Isoterma Dinámica')
            ax[0].legend(loc='upper right', fontsize=10)
            ax[1].plot(times, isoterma, color='black', linestyle='--', linewidth=1.5, label='Isoterma Dinámica')
            ax[1].legend(loc='upper right', fontsize=10)

        ax[0].format(ultitle=title_main,
                     xrotation=False,
                     xformatter='concise',
                     xlocator=xlocator,
                     xminorlocator=xminorlocator,
                     ylim=ylim,
                     yticklabelloc='both',
                     ytickloc='both',
                     xticklabelsize=8,
                     suptitle=suptitle,
                     ylabel='Altitud [msnm]',
                     xlabel=xlabel)
        ax[0].colorbar(m_main, loc='r', label=label, length=0.4)

        if tipo == 'reflectividad':
            title_add = 'Velocidad de Caída'
            label_add = '[m/s]'
        elif tipo == 'velocidad':
            title_add = 'Velocidad Adicional'
            label_add = '[m/s]'
        ax[1].format(ultitle=title_add,
                     xrotation=False,
                     xformatter='concise',
                     xlocator=xlocator,
                     xminorlocator=xminorlocator,
                     ylim=ylim,
                     yticklabelloc='both',
                     ytickloc='both',
                     xticklabelsize=8)
        ax[1].colorbar(m_add, loc='r', label=label_add, length=0.4, extend='both')

        if xlim != '':
            ax[0].format(xlim=xlim)
            ax[1].format(xlim=xlim)
    plt.show()

def plot_combinado(xlim, times, heights, Ze_filtered, Vf_filtered, isoterma_ze, isoterma_vf, hora_local=False):
    """
    Grafica Reflectividad, Velocidad de Caída y sus isotermas combinadas.
    """
    # Crear figura con tres subplots
    fig, ax = pplt.subplots(nrows=3, refwidth=5, refaspect=3, sharex=True)

    # Colormap para Reflectividad
    colorhex_ze = ['#ccd8ff','#3366ff','#9fdf9f','#00b300','#ffff00','#ffcc30',
                  '#e62e00','#ff6600','#fff0e5','#c03fc0','#602060']
    cmap_ze, norm_ze = crear_colormap(colorhex_ze, bounds=np.arange(-5,50,1))

    # Colormap para Velocidad
    colorhex_vf = ['#ccd8ff','#3366ff','#9fdf9f','#00b300','#ffff00','#ffcc30',
                  '#e62e00','#ff6600','#fff0e5','#c03fc0','#602060']
    cmap_vf, norm_vf = crear_colormap(colorhex_vf, bounds=np.arange(-10,15,1))

    xlabel = r'Hora Local $\rightarrow$' if hora_local else r'Hora UTC $\rightarrow$'
    ylim = [0, 3600] if heights[-1] < 5000 else [0, 8000]

    total_seconds = (xlim[1] - xlim[0]).total_seconds()
    if total_seconds <= 14400:
        xlocator, xminorlocator = ('hour', range(0,24,1)), ('minute',30)
    elif 14400 < total_seconds <= 82800.0:
        xlocator, xminorlocator = ('hour', range(0,24,3)), ('hour', range(0,24,1))
    else:
        xlocator, xminorlocator = ('hour', range(0,24,6)), ('hour', range(0,24,2))

    # Plot Reflectividad
    mZe = ax[0].pcolormesh(times, heights, Ze_filtered, norm=norm_ze, cmap=cmap_ze, shading='auto')
    add_no_data(ax[0], times, xlim)
    if isoterma_ze is not None:
        ax[0].plot(times, isoterma_ze, color='black', linestyle='--', linewidth=1.5, label='Isoterma Reflectividad')
        ax[0].legend(loc='upper right', fontsize=10)
    ax[0].format(ultitle='Reflectividad Equivalente',
                 xrotation=False,
                 xformatter='concise',
                 xlocator=xlocator,
                 xminorlocator=xminorlocator,
                 ylim=ylim,
                 yticklabelloc='both',
                 ytickloc='both',
                 xticklabelsize=8,
                 suptitle='Radar Perfilador MRR en UOH Rancagua',
                 ylabel='Altitud [msnm]',
                 xlabel=xlabel)
    ax[0].colorbar(mZe, loc='r', label='[dBZ]', length=0.3)

    # Plot Velocidad de Caída
    mVf = ax[1].pcolormesh(times, heights, Vf_filtered, norm=norm_vf, cmap=cmap_vf, shading='auto')
    add_no_data(ax[1], times, xlim)
    if isoterma_vf is not None:
        ax[1].plot(times, isoterma_vf, color='black', linestyle='--', linewidth=1.5, label='Isoterma Velocidad')
        ax[1].legend(loc='upper right', fontsize=10)
    ax[1].format(ultitle='Velocidad de Caída',
                 xrotation=False,
                 xformatter='concise',
                 xlocator=xlocator,
                 xminorlocator=xminorlocator,
                 ylim=ylim,
                 yticklabelloc='both',
                 ytickloc='both',
                 xticklabelsize=8,
                 ylabel='Altitud [msnm]',
                 xlabel=xlabel)
    ax[1].colorbar(mVf, loc='r', label='[m/s]', length=0.3)

    # Plot Isotermas Combinadas
    # Calculamos la isoterma combinada suavizada utilizando un método de mezcla suave
    # Por ejemplo, podemos utilizar una interpolación lineal donde ambos isoterma existen
    combined_isoterma = np.where(~np.isnan(isoterma_ze) & ~np.isnan(isoterma_vf),
                                 (isoterma_ze + isoterma_vf) / 2,
                                 np.nan)

    mCombined = ax[2].pcolormesh(times, heights, (Ze_filtered + Vf_filtered)/2, norm=norm_ze, cmap=cmap_ze, shading='auto')
    add_no_data(ax[2], times, xlim)
    if combined_isoterma is not None:
        ax[2].plot(times, combined_isoterma, color='purple', linestyle='-', linewidth=1.5, label='Isoterma Combinada')
        ax[2].legend(loc='upper right', fontsize=10)
    ax[2].format(ultitle='Reflectividad y Velocidad Combinadas',
                 xrotation=False,
                 xformatter='concise',
                 xlocator=xlocator,
                 xminorlocator=xminorlocator,
                 ylim=ylim,
                 yticklabelloc='both',
                 ytickloc='both',
                 xticklabelsize=8,
                 ylabel='Altitud [msnm]',
                 xlabel=xlabel)
    ax[2].colorbar(mCombined, loc='r', label='Combinado [dBZ/m/s]', length=0.3)

    if xlim != '':
        for axis in ax:
            axis.format(xlim=xlim)

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------
# 3) LECTURA DE DATOS
# ----------------------------------------------------------------------------
# Asegúrate de ajustar las rutas de los archivos según corresponda
ruta_reflectividad = '/home/rguerrero/MRR_analysis/datos/0802_mrr2c.nc_2.nc'   ###################################### PATH AQUI! ######################################
ruta_velocidad = ruta_reflectividad

# Cargar datos de Reflectividad
ds_ze = xr.open_dataset(ruta_reflectividad)
new_time_ze = pd.to_datetime(ds_ze.time.values)
xlim_ze = [new_time_ze[0], new_time_ze[-1]]

heights_ze = ds_ze.height[0,:].values + 500   # .values -> convertimos a np.array
Ze = ds_ze['attenuated_radar_reflectivity'].T.values
Vf_ze = ds_ze['fall_velocity'].T.values

# Cargar datos de Velocidad
ds_vf = xr.open_dataset(ruta_velocidad)
new_time_vf = pd.to_datetime(ds_vf.time.values)
xlim_vf = [new_time_vf[0], new_time_vf[-1]]

heights_vf = ds_vf.height[0,:].values + 500   # .values -> convertimos a np.array
Vf_main = ds_vf['fall_velocity'].T.values  # Usamos Vf como dato principal

# Verificar que los tiempos y alturas coincidan
assert np.array_equal(new_time_ze, new_time_vf), "Los tiempos entre reflectividad y velocidad no coinciden."
assert np.array_equal(heights_ze, heights_vf), "Las alturas entre reflectividad y velocidad no coinciden."

new_time = new_time_ze
xlim = xlim_ze
heights = heights_ze
Ze = Ze
Vf_main = Vf_main

# ----------------------------------------------------------------------------
# 4) PROCESAMIENTO DE REFLECTIVIDAD
# ----------------------------------------------------------------------------
# Graficar sin filtrar
plot_mrr(xlim, new_time, heights, {'main': Ze, 'additional': Vf_ze}, tipo='reflectividad', hora_local=False)

# Aplicar filtro mínimo
minimo_ze = 10                                                                           ##################### FILTRO DE RUIDO PARA LA REFLECTIVIDAD #########################
Ze_filtered = np.where(Ze >= minimo_ze, Ze, minimo_ze)

plot_mrr(xlim, new_time, heights, {'main': Ze_filtered, 'additional': Vf_ze}, tipo='reflectividad', hora_local=False)

# Calcular el gradiente
marco = 1
gradiente_Ze = calcular_gradiente(Ze_filtered, marco)
altura_inicial_desfase = 500 + (heights[1] - heights[0]) / 2
heights_ajustado = ds_ze.height[0,:].values + altura_inicial_desfase

# Graficar gradiente de Reflectividad
plot_mrr(xlim, new_time, heights_ajustado, {'main': gradiente_Ze}, tipo='reflectividad', hora_local=False)

# Primera iteración - rango fijo
hmin_ini, hmax_ini = 100, 5000
res_isoterma_1_ze = detectar_isoterma_cero_acotada(gradiente_Ze, heights_ajustado, hmin=hmin_ini, hmax=hmax_ini)

grad_mod_1_ze = np.copy(gradiente_Ze)
IsotermaAlturas_1_ze = []  # aquí guardamos la altura real
for r in res_isoterma_1_ze:
    t = r["iter"]
    alt_min = r["altura_minima"]  # la altura real o NaN
    if not np.isnan(alt_min):
        idx_alt = int(np.abs(heights_ajustado - alt_min).argmin())
        IsotermaAlturas_1_ze.append(alt_min)
        grad_mod_1_ze[idx_alt, t] = np.nan  # Usamos NaN en lugar de un valor fijo
    else:
        IsotermaAlturas_1_ze.append(np.nan)

plot_mrr(xlim, new_time, heights_ajustado, {'main': grad_mod_1_ze}, tipo='reflectividad', hora_local=False)

# Calcular media y std ignorando NaNs
mean_iso_1_ze = np.nanmean(IsotermaAlturas_1_ze)
std_iso_1_ze  = np.nanstd(IsotermaAlturas_1_ze)

hmin_dyn_ze = mean_iso_1_ze - std_iso_1_ze -200
hmax_dyn_ze = mean_iso_1_ze + std_iso_1_ze +200

print(f"Reflectividad - Primera iteración - Mean: {mean_iso_1_ze:.1f}, Std: {std_iso_1_ze:.1f}")
print(f"Reflectividad - Nuevos límites: [hmin_dyn={hmin_dyn_ze:.1f}, hmax_dyn={hmax_dyn_ze:.1f}]")

# Segunda iteración - rango dinámico
res_isoterma_2_ze = detectar_isoterma_cero_acotada(gradiente_Ze, heights_ajustado, hmin=hmin_dyn_ze, hmax=hmax_dyn_ze)

grad_mod_2_ze = np.copy(gradiente_Ze)
IsotermaAlturas_2_ze = []
for r in res_isoterma_2_ze:
    t = r["iter"]
    alt_min = r["altura_minima"]
    if not np.isnan(alt_min):
        idx_alt = int(np.abs(heights_ajustado - alt_min).argmin())
        IsotermaAlturas_2_ze.append(alt_min)
        grad_mod_2_ze[idx_alt, t] = np.nan  # Usamos NaN en lugar de un valor fijo
    else:
        IsotermaAlturas_2_ze.append(np.nan)

plot_mrr(xlim, new_time, heights_ajustado, {'main': grad_mod_2_ze}, tipo='reflectividad', hora_local=False)

# Suavizado
IsotermaAlturas_2_ze = np.array(IsotermaAlturas_2_ze)
Isoterma_suav_2_ze = suavizado_media_movil(IsotermaAlturas_2_ze, ventana=3)  # Ajusta la ventana según sea necesario
Isoterma_suav_2_ze[np.isnan(IsotermaAlturas_2_ze)] = np.nan

Isoterma_loess_2_ze  = filtrar_isoterma_loess(
    Isoterma_suav_2_ze,
    x=np.arange(len(Isoterma_suav_2_ze)),
    frac=0.17,
    it=3
)

# ----------------------------------------------------------------------------
# 5) PROCESAMIENTO DE VELOCIDAD
# ----------------------------------------------------------------------------
# Graficar sin filtrar
plot_mrr(xlim, new_time, heights, {'main': Vf_main}, tipo='velocidad', hora_local=False)

# Aplicar filtro mínimo
minimo_vf = -5                                                                     ##################### FILTRO DE RUIDO PARA LA VELOCIDAD #########################
Vf_filtered = np.where(Vf_main >= minimo_vf, Vf_main, minimo_vf)

plot_mrr(xlim, new_time, heights, {'main': Vf_filtered}, tipo='velocidad', hora_local=False)

# Calcular el gradiente
gradiente_Vf = calcular_gradiente(Vf_filtered, marco)
# heights_ajustado ya está definido anteriormente

# Graficar gradiente de Velocidad
plot_mrr(xlim, new_time, heights_ajustado, {'main': gradiente_Vf}, tipo='velocidad', hora_local=False)

# Primera iteración - rango fijo
hmin_ini_vf, hmax_ini_vf = 100, 5000
res_isoterma_1_vf = detectar_isoterma_cero_acotada(gradiente_Vf, heights_ajustado, hmin=hmin_ini_vf, hmax=hmax_ini_vf)

grad_mod_1_vf = np.copy(gradiente_Vf)
IsotermaAlturas_1_vf = []
for r in res_isoterma_1_vf:
    t = r["iter"]
    alt_min = r["altura_minima"]
    if not np.isnan(alt_min):
        idx_alt = int(np.abs(heights_ajustado - alt_min).argmin())
        IsotermaAlturas_1_vf.append(alt_min)
        grad_mod_1_vf[idx_alt, t] = np.nan
    else:
        IsotermaAlturas_1_vf.append(np.nan)

plot_mrr(xlim, new_time, heights_ajustado, {'main': grad_mod_1_vf}, tipo='velocidad', hora_local=False)

# Calcular media y std ignorando NaNs
mean_iso_1_vf = np.nanmean(IsotermaAlturas_1_vf)
std_iso_1_vf  = np.nanstd(IsotermaAlturas_1_vf)

hmin_dyn_vf = mean_iso_1_vf - std_iso_1_vf -200
hmax_dyn_vf = mean_iso_1_vf + std_iso_1_vf +200

print(f"Velocidad - Primera iteración - Mean: {mean_iso_1_vf:.1f}, Std: {std_iso_1_vf:.1f}")
print(f"Velocidad - Nuevos límites: [hmin_dyn={hmin_dyn_vf:.1f}, hmax_dyn={hmax_dyn_vf:.1f}]")

# Segunda iteración - rango dinámico
res_isoterma_2_vf = detectar_isoterma_cero_acotada(gradiente_Vf, heights_ajustado, hmin=hmin_dyn_vf, hmax=hmax_dyn_vf)

grad_mod_2_vf = np.copy(gradiente_Vf)
IsotermaAlturas_2_vf = []
for r in res_isoterma_2_vf:
    t = r["iter"]
    alt_min = r["altura_minima"]
    if not np.isnan(alt_min):
        idx_alt = int(np.abs(heights_ajustado - alt_min).argmin())
        IsotermaAlturas_2_vf.append(alt_min)
        grad_mod_2_vf[idx_alt, t] = np.nan
    else:
        IsotermaAlturas_2_vf.append(np.nan)

plot_mrr(xlim, new_time, heights_ajustado, {'main': grad_mod_2_vf}, tipo='velocidad', hora_local=False)

# Suavizado
IsotermaAlturas_2_vf = np.array(IsotermaAlturas_2_vf)
Isoterma_suav_2_vf = suavizado_media_movil(IsotermaAlturas_2_vf, ventana=3)  # Ajusta la ventana según sea necesario
Isoterma_suav_2_vf[np.isnan(IsotermaAlturas_2_vf)] = np.nan

Isoterma_loess_2_vf  = filtrar_isoterma_loess(
    Isoterma_suav_2_vf,
    x=np.arange(len(Isoterma_suav_2_vf)),
    frac=0.17,
    it=3
)

# ----------------------------------------------------------------------------
# 6) COMPARACIÓN VISUAL PRIMERA vs SEGUNDA ITERACIÓN
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(IsotermaAlturas_1_ze, label="Isoterma 1 Reflectividad (Fijo)", 
        linestyle='--', color='blue', linewidth=2)
ax.plot(Isoterma_loess_2_ze, label="Isoterma 2 Reflectividad (Dinámico + RLOESS)",
        linestyle='-', color='red', linewidth=2)
ax.plot(IsotermaAlturas_1_vf, label="Isoterma 1 Velocidad (Fijo)", 
        linestyle='--', color='green', linewidth=2)
ax.plot(Isoterma_loess_2_vf, label="Isoterma 2 Velocidad (Dinámico + RLOESS)",
        linestyle='-', color='orange', linewidth=2)
ax.set_xlabel('Tiempo (s)', fontsize=12)
ax.set_ylabel('Altitud [msnm]', fontsize=12)
ax.set_title('Comparación Isotermas - Iteración Fija vs Iteración Dinámica', fontsize=14)
ax.set_xlim([0, len(IsotermaAlturas_1_ze)])
ax.set_ylim([0, heights[-1]])
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
ax.legend(loc='upper right', fontsize=10)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 7) PLOTEO FINAL COMBINADO
# ----------------------------------------------------------------------------
# Ajustar la longitud de las isotermas si difiere del tiempo
def ajustar_longitud(isoterma, tiempo):
    if len(isoterma) < len(tiempo):
        dif = len(tiempo) - len(isoterma)
        isoterma = np.pad(isoterma, (0, dif), 'edge')
    elif len(isoterma) > len(tiempo):
        isoterma = isoterma[:len(tiempo)]
    return isoterma

Isoterma_loess_2_ze = ajustar_longitud(Isoterma_loess_2_ze, new_time)
Isoterma_loess_2_vf = ajustar_longitud(Isoterma_loess_2_vf, new_time)

# Plot final combinado
plot_combinado(
    xlim=xlim,
    times=new_time,
    heights=heights,
    Ze_filtered=Ze_filtered,
    Vf_filtered=Vf_filtered,
    isoterma_ze=Isoterma_loess_2_ze,
    isoterma_vf=Isoterma_loess_2_vf,
    hora_local=False
)

print("Proceso completado con doble iteración para Reflectividad y Velocidad.",
      "\nSe han calculado y suavizado las isotermas para ambos datasets.",
      "\nEl gráfico final muestra Reflectividad, Velocidad de Caída y su combinación con isotermas suavizadas.",
      "\nSe han excluido las isotermas no válidas de los cálculos de promedio para evitar distorsiones.")
