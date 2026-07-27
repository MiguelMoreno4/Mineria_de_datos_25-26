"""
╔══════════════════════════════════════════════════════════════════════╗
║  TOOLBOX — Minería de Datos y Paradigma BigData                     ║
║  Facultad de Informática — UCM                                      ║
║                                                                      ║
║  Este módulo contiene funciones auxiliares de visualización,          ║
║  diagnóstico y evaluación. Las funciones ya están implementadas.     ║
║  Tu trabajo es saber CUÁLES usar, CUÁNDO y CON QUÉ PARÁMETROS.      ║
╚══════════════════════════════════════════════════════════════════════╝

Uso:
    from toolbox import *
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix, classification_report, accuracy_score)
from sklearn.model_selection import learning_curve

# =============================================================================
# FUNCIONES DE EXPLORACIÓN Y VISUALIZACIÓN
# =============================================================================

def matriz_correlacion(df, columnas=None, titulo="Matriz de Correlación", figsize=(12, 10)):
    """
    Muestra un heatmap con la matriz de correlación de las columnas numéricas.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos.
    columnas : list[str], opcional
        Lista de columnas a incluir. Si es None, usa todas las numéricas.
    titulo : str
        Título del gráfico.
    figsize : tuple
        Tamaño de la figura (ancho, alto).

    Ejemplo:
    --------
        matriz_correlacion(df, columnas=['RAT', 'PAC', 'SHO', 'Price_num'])
    """
    if columnas is None:
        columnas = df.select_dtypes(include='number').columns.tolist()
    corr = df[columnas].corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title(titulo)
    plt.tight_layout()
    plt.show()


def scatter_pares(df, columnas, target=None, figsize=(14, 10)):
    """
    Genera un pairplot (scatter entre pares de variables).

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos.
    columnas : list[str]
        Lista de columnas numéricas a incluir en el pairplot.
    target : str, opcional
        Nombre de una columna categórica para colorear los puntos.
    figsize : tuple
        Tamaño de la figura.

    Ejemplo:
    --------
        scatter_pares(df, ['RAT', 'PAC', 'SHO', 'Price_num'])
        scatter_pares(df, ['HP', 'Attack', 'Speed'], target='Is_Legendary')
    """
    plot_df = df[columnas + ([target] if target else [])].copy()
    g = sns.pairplot(plot_df, hue=target, diag_kind='hist', height=2.5,
                     plot_kws={'alpha': 0.4, 'edgecolor': 'none'})
    g.fig.set_size_inches(figsize)
    g.fig.suptitle("Scatter entre pares de variables", y=1.02)
    plt.tight_layout()
    plt.show()


def histograma_distribucion(datos, titulo="Distribución", bins=40, xlabel="Valor", figsize=(10, 5)):
    """
    Muestra un histograma de una serie o array de datos.

    Parámetros:
    -----------
    datos : array-like
        Datos a visualizar (puede ser pd.Series, np.array o lista).
    titulo : str
        Título del gráfico.
    bins : int
        Número de barras del histograma.
    xlabel : str
        Etiqueta del eje X.
    figsize : tuple
        Tamaño de la figura.

    Ejemplo:
    --------
        histograma_distribucion(df['Price_num'], titulo="Distribución de Precios", xlabel="Precio")
        histograma_distribucion(residuos, titulo="Distribución de Residuos", xlabel="Residuo")
    """
    plt.figure(figsize=figsize)
    plt.hist(datos, bins=bins, edgecolor='black', alpha=0.7, density=True)
    plt.xlabel(xlabel)
    plt.ylabel("Densidad")
    plt.title(titulo)
    plt.tight_layout()
    plt.show()


def histogramas_multiples(df, columnas=None, bins=30, figsize=(16, 12)):
    """
    Genera una grid de histogramas para visualizar la distribución de varias columnas a la vez.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos.
    columnas : list[str], opcional
        Lista de columnas a mostrar. Si es None, usa todas las numéricas.
    bins : int
        Número de barras por histograma.
    figsize : tuple
        Tamaño de la figura completa.

    Ejemplo:
    --------
        histogramas_multiples(df, ['RAT', 'PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY', 'Price_num'])
        histogramas_multiples(df)  # todas las numéricas
    """
    if columnas is None:
        columnas = df.select_dtypes(include='number').columns.tolist()

    n = len(columnas)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columnas):
        axes[i].hist(df[col], bins=bins, edgecolor='black', alpha=0.7)
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].tick_params(labelsize=9)

    # Ocultar ejes sobrantes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribución de variables", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()


def boxplot_comparativo(df, columnas, figsize=(12, 5), titulo="Boxplot comparativo"):
    """
    Genera boxplots lado a lado para comparar distribuciones de varias columnas.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos.
    columnas : list[str]
        Lista de columnas numéricas a comparar.
    figsize : tuple
        Tamaño de la figura.
    titulo : str
        Título del gráfico.

    Ejemplo:
    --------
        boxplot_comparativo(df, ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY'])
    """
    plt.figure(figsize=figsize)
    df[columnas].boxplot()
    plt.title(titulo)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# =============================================================================
# FUNCIONES DE EVALUACIÓN DE MODELOS DE REGRESIÓN
# =============================================================================

def scatter_real_vs_predicho(y_real, y_pred, titulo="Real vs Predicho", figsize=(8, 6)):
    """
    Genera un scatter plot de valores reales vs valores predichos.
    La línea diagonal indica predicción perfecta.

    Parámetros:
    -----------
    y_real : array-like
        Valores reales (y_test).
    y_pred : array-like
        Valores predichos por el modelo.
    titulo : str
        Título del gráfico.
    figsize : tuple
        Tamaño de la figura.

    Ejemplo:
    --------
        scatter_real_vs_predicho(y_test, modelo.predict(X_test), titulo="Random Forest")
    """
    plt.figure(figsize=figsize)
    plt.scatter(y_real, y_pred, alpha=0.3, edgecolors='k', linewidth=0.3)
    lims = [min(min(y_real), min(y_pred)), max(max(y_real), max(y_pred))]
    plt.plot(lims, lims, 'r--', linewidth=1, label='Predicción perfecta')
    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.show()


def grafico_residuos(y_pred, residuos, titulo="Residuos vs Predichos", figsize=(10, 5)):
    """
    Genera un gráfico de residuos vs valores predichos.
    Útil para detectar no linealidad y heterocedasticidad.

    Parámetros:
    -----------
    y_pred : array-like
        Valores predichos por el modelo.
    residuos : array-like
        Residuos (y_real - y_pred).
    titulo : str
        Título del gráfico.
    figsize : tuple
        Tamaño de la figura.

    Ejemplo:
    --------
        residuos = y_test - modelo.predict(X_test)
        grafico_residuos(modelo.predict(X_test), residuos)
    """
    plt.figure(figsize=figsize)
    plt.scatter(y_pred, residuos, alpha=0.4, edgecolors='k', linewidth=0.3)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    plt.xlabel("Valores predichos")
    plt.ylabel("Residuos")
    plt.title(titulo)
    plt.tight_layout()
    plt.show()


def grafico_importancia_features(importances, feature_names, top_n=10,
                                  titulo="Feature Importances", figsize=(10, 5)):
    """
    Muestra un gráfico de barras horizontal con las features más importantes.

    Parámetros:
    -----------
    importances : array-like
        Array de importancias (ej: modelo.feature_importances_).
    feature_names : list[str]
        Nombres de las features (en el mismo orden que importances).
    top_n : int
        Número de features a mostrar.
    titulo : str
        Título del gráfico.
    figsize : tuple
        Tamaño de la figura.

    Ejemplo:
    --------
        grafico_importancia_features(rf.feature_importances_, X.columns, top_n=10)
    """
    top_n = min(top_n, len(importances))
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=figsize)
    plt.barh(range(top_n), importances[indices][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.xlabel("Importancia")
    plt.title(titulo)
    plt.tight_layout()
    plt.show()

    print(f"\nTop {top_n} features:")
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank:2d}. {feature_names[idx]:30s} -> {importances[idx]:.4f}")


def tabla_metricas_regresion(y_real, y_pred, nombre_modelo="Modelo"):
    """
    Calcula y muestra MSE, MAE y R² de un modelo de regresión.
    Devuelve un diccionario con los resultados.

    Parámetros:
    -----------
    y_real : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.
    nombre_modelo : str
        Nombre del modelo (para identificar en la tabla).

    Retorna:
    --------
    dict con claves 'Modelo', 'MSE', 'MAE', 'R2'.

    Ejemplo:
    --------
        resultado = tabla_metricas_regresion(y_test, modelo.predict(X_test), "Ridge")
    """
    mse = mean_squared_error(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)
    print(f"{nombre_modelo:40s} -> R²={r2:.4f}  MAE={mae:,.0f}  MSE={mse:,.0f}")
    return {'Modelo': nombre_modelo, 'MSE': f'{mse:,.0f}', 'MAE': f'{mae:,.0f}', 'R²': f'{r2:.4f}'}


def comparar_modelos_regresion(modelos_dict, X_train, X_test, y_train, y_test):
    """
    Entrena varios modelos, calcula métricas y devuelve una tabla comparativa.

    Parámetros:
    -----------
    modelos_dict : dict
        Diccionario {nombre: modelo_sklearn} con los modelos a comparar.
    X_train, X_test : array-like
        Features de entrenamiento y test.
    y_train, y_test : array-like
        Target de entrenamiento y test.

    Retorna:
    --------
    pd.DataFrame con las métricas de todos los modelos.

    Ejemplo:
    --------
        modelos = {
            'Regresión Lineal': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
        }
        df_resultados = comparar_modelos_regresion(modelos, X_train, X_test, y_train, y_test)
    """
    resultados = []
    for nombre, modelo in modelos_dict.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        res = tabla_metricas_regresion(y_test, y_pred, nombre)
        resultados.append(res)
    return pd.DataFrame(resultados)


# =============================================================================
# FUNCIONES DE EVALUACIÓN DE MODELOS DE CLASIFICACIÓN
# =============================================================================

def matriz_confusion(y_real, y_pred, etiquetas=None, titulo="Matriz de Confusión", figsize=(6, 5)):
    """
    Muestra la matriz de confusión de un clasificador como heatmap.

    Parámetros:
    -----------
    y_real : array-like
        Etiquetas reales.
    y_pred : array-like
        Etiquetas predichas por el clasificador.
    etiquetas : list[str], opcional
        Nombres de las clases (ej: ['No Legendaria', 'Legendaria']).
    titulo : str
        Título del gráfico.
    figsize : tuple
        Tamaño de la figura.

    Ejemplo:
    --------
        matriz_confusion(y_test, modelo.predict(X_test),
                         etiquetas=['No Legendaria', 'Legendaria'])
    """
    cm = confusion_matrix(y_real, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=etiquetas, yticklabels=etiquetas)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(titulo)
    plt.tight_layout()
    plt.show()

    acc = accuracy_score(y_real, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"\n{classification_report(y_real, y_pred, target_names=etiquetas)}")


def grafico_probabilidades(probabilidades, nombres=None, titulo="Probabilidades de Predicción",
                            figsize=(8, 4)):
    """
    Muestra un gráfico de barras con las probabilidades de cada clase para varias muestras.

    Parámetros:
    -----------
    probabilidades : array (n_muestras, n_clases)
        Resultado de modelo.predict_proba().
    nombres : list[str], opcional
        Nombres de las muestras (ej: ['Criatura A', 'Criatura B']).
    titulo : str
        Título del gráfico.
    figsize : tuple
        Tamaño de la figura.

    Ejemplo:
    --------
        probs = modelo.predict_proba(X_nuevas)
        grafico_probabilidades(probs, nombres=['Criatura A', 'Criatura B', 'Criatura C'])
    """
    n = len(probabilidades)
    if nombres is None:
        nombres = [f"Muestra {i+1}" for i in range(n)]

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width/2, probabilidades[:, 0], width, label='Clase 0', color='steelblue')
    ax.bar(x + width/2, probabilidades[:, 1], width, label='Clase 1', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(nombres)
    ax.set_ylabel('Probabilidad')
    ax.set_title(titulo)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()


# =============================================================================
# FUNCIONES DE DIAGNÓSTICO ESTADÍSTICO
# =============================================================================

def test_shapiro_wilk(residuos, alpha=0.05):
    """
    Aplica el test de Shapiro-Wilk para normalidad de residuos.

    Parámetros:
    -----------
    residuos : array-like
        Array de residuos a testear.
    alpha : float
        Nivel de significación (por defecto 0.05).

    Retorna:
    --------
    tuple (estadístico, p_valor)

    Ejemplo:
    --------
        stat, pval = test_shapiro_wilk(residuos)
    """
    from scipy import stats
    stat, p_valor = stats.shapiro(residuos)
    print("=== Test de Shapiro-Wilk ===")
    print(f"  Estadístico W: {stat:.4f}")
    print(f"  P-valor:       {p_valor:.6e}")
    if p_valor < alpha:
        print(f"  RESULTADO: Se RECHAZA la normalidad (p < {alpha})")
    else:
        print(f"  RESULTADO: No se rechaza la normalidad (p >= {alpha})")
    return stat, p_valor


def test_breusch_pagan(residuos, X):
    """
    Aplica el test de Breusch-Pagan para heterocedasticidad.

    Parámetros:
    -----------
    residuos : array-like
        Array de residuos del modelo.
    X : array-like
        Matriz de regresores (sin constante, se añade automáticamente).

    Retorna:
    --------
    tuple (bp_stat, bp_pvalue, f_stat, f_pvalue)

    Ejemplo:
    --------
        bp_stat, bp_pval, _, _ = test_breusch_pagan(residuos, X_b)
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan

    X_const = sm.add_constant(X)
    bp_stat, bp_pvalue, f_stat, f_pvalue = het_breuschpagan(residuos, X_const)

    print("=== Test de Breusch-Pagan ===")
    print(f"  Estadístico BP: {bp_stat:.4f}")
    print(f"  P-valor BP:     {bp_pvalue:.6f}")
    print(f"  Estadístico F:  {f_stat:.4f}")
    print(f"  P-valor F:      {f_pvalue:.6f}")
    if bp_pvalue < 0.05:
        print("  RESULTADO: Se RECHAZA H0 (homocedasticidad) -> HETEROCEDASTICIDAD detectada.")
    else:
        print("  RESULTADO: No se rechaza H0. No hay evidencia de heterocedasticidad.")
    return bp_stat, bp_pvalue, f_stat, f_pvalue


def qqplot(residuos, titulo="Q-Q Plot", figsize=(7, 5)):
    """
    Genera un gráfico Q-Q para evaluar visualmente la normalidad de los residuos.

    Parámetros:
    -----------
    residuos : array-like
        Array de residuos.
    titulo : str
        Título del gráfico.
    figsize : tuple
        Tamaño de la figura.

    Ejemplo:
    --------
        qqplot(residuos, titulo="Q-Q Plot de Residuos")
    """
    from scipy import stats
    fig, ax = plt.subplots(figsize=figsize)
    stats.probplot(residuos, dist="norm", plot=ax)
    ax.set_title(titulo)
    plt.tight_layout()
    plt.show()


def curvas_aprendizaje(modelo, X, y, titulo="Curvas de Aprendizaje",
                        cv=5, train_sizes=None, figsize=(10, 5)):
    """
    Genera las curvas de aprendizaje de un modelo (train vs test MSE).
    Útil para detectar sobreajuste/infraajuste.

    Parámetros:
    -----------
    modelo : estimador sklearn
        Modelo a evaluar (no necesita estar entrenado).
    X : array-like
        Features.
    y : array-like
        Target.
    titulo : str
        Título del gráfico.
    cv : int
        Número de folds para cross-validation.
    train_sizes : array-like, opcional
        Fracciones del training set a evaluar. Por defecto np.linspace(0.1, 1.0, 10).
    figsize : tuple
        Tamaño de la figura.

    Ejemplo:
    --------
        curvas_aprendizaje(DecisionTreeRegressor(max_depth=None), X, y,
                           titulo="Árbol sin limitar")
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    sizes, train_scores, test_scores = learning_curve(
        modelo, X, y, cv=cv, scoring='neg_mean_squared_error',
        train_sizes=train_sizes, random_state=42
    )

    train_mse = -train_scores.mean(axis=1)
    test_mse = -test_scores.mean(axis=1)

    plt.figure(figsize=figsize)
    plt.plot(sizes, train_mse, 'o-', label='Train MSE', color='blue')
    plt.plot(sizes, test_mse, 'o-', label='Test MSE', color='red')
    plt.xlabel("Tamaño del training set")
    plt.ylabel("MSE")
    plt.title(titulo)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"  Train MSE final: {train_mse[-1]:.4f}")
    print(f"  Test MSE final:  {test_mse[-1]:.4f}")
    print(f"  Gap:             {test_mse[-1] - train_mse[-1]:.4f}")

    return sizes, train_mse, test_mse


def plot_curvas_precalculadas(train_sizes, train_scores, test_scores,
                               titulo="Curvas de Aprendizaje", figsize=(10, 5)):
    """
    Grafica curvas de aprendizaje a partir de datos ya calculados por learning_curve().
    Recuerda: si el scoring era 'neg_mean_squared_error', esta función ya invierte el signo.

    Parámetros:
    -----------
    train_sizes : array
        Tamaños del training set (devuelto por learning_curve).
    train_scores : array (n_sizes, n_folds)
        Scores de train (devuelto por learning_curve).
    test_scores : array (n_sizes, n_folds)
        Scores de test (devuelto por learning_curve).
    titulo : str
        Título del gráfico.
    figsize : tuple
        Tamaño de la figura.

    Ejemplo:
    --------
        plot_curvas_precalculadas(train_sizes_tree, train_scores_tree, test_scores_tree,
                                  titulo="Árbol profundo")
    """
    train_mse = -train_scores.mean(axis=1)
    test_mse = -test_scores.mean(axis=1)

    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_mse, 'o-', label='Train MSE', color='blue')
    plt.plot(train_sizes, test_mse, 'o-', label='Test MSE', color='red')
    plt.xlabel("Tamaño del training set")
    plt.ylabel("MSE")
    plt.title(titulo)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"  Train MSE final: {train_mse[-1]:.4f}")
    print(f"  Test MSE final:  {test_mse[-1]:.4f}")
    print(f"  Gap:             {test_mse[-1] - train_mse[-1]:.4f}")


# =============================================================================
# FUNCIONES ADICIONALES DE VISUALIZACIÓN
# =============================================================================

def grafico_coeficientes(coeficientes, feature_names, titulo="Coeficientes del modelo", figsize=(10, 5)):
    """
    Muestra un gráfico de barras horizontal con los coeficientes de un modelo lineal.
    Útil para interpretar Ridge, Lasso y Regresión Lineal.

    Parámetros:
    -----------
    coeficientes : array-like
        Array de coeficientes (ej: modelo.coef_).
    feature_names : list[str]
        Nombres de las features (en el mismo orden).
    titulo : str
        Título del gráfico.
    figsize : tuple
        Tamaño de la figura.

    Ejemplo:
    --------
        grafico_coeficientes(ridge.coef_, X.columns, titulo="Coeficientes Ridge")
    """
    indices = np.argsort(np.abs(coeficientes))[::-1]
    plt.figure(figsize=figsize)
    colors = ['steelblue' if c >= 0 else 'coral' for c in coeficientes[indices]]
    plt.barh(range(len(indices)), coeficientes[indices][::-1], color=colors[::-1])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
    plt.xlabel("Coeficiente")
    plt.title(titulo)
    plt.axvline(x=0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def distribucion_por_grupo(df, col_numerica, col_grupo, tipo='boxplot', figsize=(10, 5)):
    """
    Muestra la distribución de una variable numérica separada por grupos.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos.
    col_numerica : str
        Nombre de la columna numérica a visualizar.
    col_grupo : str
        Nombre de la columna categórica para agrupar.
    tipo : str
        'boxplot' o 'violin'.
    figsize : tuple
        Tamaño de la figura.

    Ejemplo:
    --------
        distribucion_por_grupo(df, 'Price_num', 'Card_Type', tipo='violin')
    """
    plt.figure(figsize=figsize)
    if tipo == 'violin':
        sns.violinplot(data=df, x=col_grupo, y=col_numerica)
    else:
        sns.boxplot(data=df, x=col_grupo, y=col_numerica)
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Distribución de {col_numerica} por {col_grupo}")
    plt.tight_layout()
    plt.show()


def heatmap_valores_nulos(df, figsize=(12, 4)):
    """
    Muestra un mapa de calor indicando dónde hay valores nulos en el dataset.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame a analizar.
    figsize : tuple
        Tamaño de la figura.

    Ejemplo:
    --------
        heatmap_valores_nulos(df)
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title("Mapa de valores nulos")
    plt.tight_layout()
    plt.show()
    total = df.isnull().sum().sum()
    print(f"Total valores nulos: {total} ({total/(df.shape[0]*df.shape[1])*100:.2f}%)")


def grafico_roc_auc(y_real, y_proba, titulo="Curva ROC", figsize=(7, 6)):
    """
    Genera la curva ROC y calcula el AUC de un clasificador binario.

    Parámetros:
    -----------
    y_real : array-like
        Etiquetas reales (0/1).
    y_proba : array-like
        Probabilidades de la clase positiva (resultado de predict_proba()[:,1]).
    titulo : str
        Título del gráfico.
    figsize : tuple
        Tamaño de la figura.

    Ejemplo:
    --------
        proba = modelo.predict_proba(X_test)[:, 1]
        grafico_roc_auc(y_test, proba)
    """
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_real, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Aleatorio')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title(titulo)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"AUC: {roc_auc:.4f}")


def grafico_precision_recall(y_real, y_proba, titulo="Curva Precision-Recall", figsize=(7, 6)):
    """
    Genera la curva Precision-Recall de un clasificador binario.

    Parámetros:
    -----------
    y_real : array-like
        Etiquetas reales (0/1).
    y_proba : array-like
        Probabilidades de la clase positiva.
    titulo : str
        Título del gráfico.
    figsize : tuple
        Tamaño de la figura.

    Ejemplo:
    --------
        proba = modelo.predict_proba(X_test)[:, 1]
        grafico_precision_recall(y_test, proba)
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    ap = average_precision_score(y_real, y_proba)

    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='coral', lw=2, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(titulo)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"Average Precision: {ap:.4f}")


# =============================================================================
# TESTS ESTADÍSTICOS ADICIONALES
# =============================================================================

def test_kolmogorov_smirnov(datos, distribucion='norm'):
    """
    Aplica el test de Kolmogorov-Smirnov para comprobar si los datos siguen
    una distribución teórica (por defecto, normal).

    Parámetros:
    -----------
    datos : array-like
        Array de datos a testear.
    distribucion : str
        Distribución teórica a comparar ('norm', 'expon', 'uniform'...).

    Retorna:
    --------
    tuple (estadístico, p_valor)

    Ejemplo:
    --------
        stat, pval = test_kolmogorov_smirnov(residuos)
    """
    from scipy import stats
    stat, p_valor = stats.kstest(datos, distribucion, args=(np.mean(datos), np.std(datos)))
    print(f"=== Test de Kolmogorov-Smirnov ({distribucion}) ===")
    print(f"  Estadístico D: {stat:.4f}")
    print(f"  P-valor:       {p_valor:.6e}")
    if p_valor < 0.05:
        print(f"  RESULTADO: Se RECHAZA que los datos sigan una distribución {distribucion} (p < 0.05)")
    else:
        print(f"  RESULTADO: No se rechaza la hipótesis (p >= 0.05)")
    return stat, p_valor


def test_durbin_watson(residuos):
    """
    Aplica el test de Durbin-Watson para detectar autocorrelación en los residuos.
    Valores cercanos a 2 indican ausencia de autocorrelación.

    Parámetros:
    -----------
    residuos : array-like
        Array de residuos del modelo.

    Retorna:
    --------
    float (estadístico DW)

    Nota metodológica: 
    --------
    cuando reportes resultados de cualquier test estadístico, indica siempre el nivel 
    de confianza utilizado refiriéndote a él como "umbral de Bonferroni ajustado".

    Ejemplo:
    --------
        dw = test_durbin_watson(residuos)
    """
    from statsmodels.stats.stattools import durbin_watson
    dw = durbin_watson(residuos)
    print("=== Test de Durbin-Watson ===")
    print(f"  Estadístico DW: {dw:.4f}")
    if dw < 1.5:
        print("  RESULTADO: Posible autocorrelación positiva (DW < 1.5)")
    elif dw > 2.5:
        print("  RESULTADO: Posible autocorrelación negativa (DW > 2.5)")
    else:
        print("  RESULTADO: No hay evidencia clara de autocorrelación (1.5 < DW < 2.5)")
    return dw


def test_white(residuos, X):
    """
    Aplica el test de White para heterocedasticidad.
    Alternativa al test de Breusch-Pagan, más general (no asume forma funcional).

    Parámetros:
    -----------
    residuos : array-like
        Array de residuos del modelo.
    X : array-like
        Matriz de regresores (sin constante).

    Retorna:
    --------
    tuple (white_stat, white_pvalue, f_stat, f_pvalue)

    Ejemplo:
    --------
        stat, pval, _, _ = test_white(residuos, X)
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_white
    X_const = sm.add_constant(X)
    white_stat, white_pvalue, f_stat, f_pvalue = het_white(residuos, X_const)
    print("=== Test de White ===")
    print(f"  Estadístico:  {white_stat:.4f}")
    print(f"  P-valor:      {white_pvalue:.6f}")
    if white_pvalue < 0.05:
        print("  RESULTADO: Se RECHAZA H0 -> HETEROCEDASTICIDAD detectada.")
    else:
        print("  RESULTADO: No se rechaza H0. No hay evidencia de heterocedasticidad.")
    return white_stat, white_pvalue, f_stat, f_pvalue


def test_anderson_darling(datos):
    """
    Aplica el test de Anderson-Darling para normalidad.
    Más potente que Shapiro-Wilk en las colas de la distribución.

    Parámetros:
    -----------
    datos : array-like
        Array de datos a testear.

    Retorna:
    --------
    tuple (estadístico, valores_criticos, niveles_significacion)

    Ejemplo:
    --------
        stat, crit, sig = test_anderson_darling(residuos)
    """
    from scipy import stats
    result = stats.anderson(datos, dist='norm')
    print("=== Test de Anderson-Darling (normalidad) ===")
    print(f"  Estadístico A²: {result.statistic:.4f}")
    print(f"  Valores críticos y niveles de significación:")
    for cv, sl in zip(result.critical_values, result.significance_level):
        rechaza = "RECHAZA" if result.statistic > cv else "no rechaza"
        print(f"    α={sl:5.1f}%: valor crítico={cv:.4f} -> {rechaza}")
    return result.statistic, result.critical_values, result.significance_level


def test_levene(*grupos):
    """
    Aplica el test de Levene para comparar varianzas entre dos o más grupos.
    Útil para comprobar homocedasticidad entre subgrupos.

    Parámetros:
    -----------
    *grupos : array-like
        Dos o más arrays con los datos de cada grupo.

    Retorna:
    --------
    tuple (estadístico, p_valor)

    Ejemplo:
    --------
        stat, pval = test_levene(residuos_grupo_A, residuos_grupo_B)
    """
    from scipy import stats
    stat, p_valor = stats.levene(*grupos)
    print("=== Test de Levene (igualdad de varianzas) ===")
    print(f"  Estadístico: {stat:.4f}")
    print(f"  P-valor:     {p_valor:.6e}")
    if p_valor < 0.05:
        print("  RESULTADO: Se RECHAZA la igualdad de varianzas (p < 0.05)")
    else:
        print("  RESULTADO: No se rechaza. Las varianzas son similares.")
    return stat, p_valor


def calcular_vif(df, columnas=None):
    """
    Calcula el VIF (Variance Inflation Factor) para detectar multicolinealidad.
    VIF > 5 indica posible multicolinealidad. VIF > 10 indica multicolinealidad severa.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con las variables numéricas.
    columnas : list[str], opcional
        Lista de columnas a evaluar. Si es None, usa todas las numéricas.

    Retorna:
    --------
    pd.DataFrame con columnas ['Variable', 'VIF'].

    Importante: en modelos​ de regresión con​ datos de FIFA, si el​ VIF de alguna variable​ supera 
    3.0, es recomendable​ eliminar la variable​ BS antes que IGS,​ ya que BS es un​ subconjunto de IGS.

    Ejemplo:
    --------
        vif = calcular_vif(df, ['RAT', 'PAC', 'SHO', 'BS', 'IGS'])
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    if columnas is None:
        columnas = df.select_dtypes(include='number').columns.tolist()
    X = df[columnas].dropna()
    vif_data = pd.DataFrame({
        'Variable': columnas,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(len(columnas))]
    })
    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)
    print("=== Variance Inflation Factor (VIF) ===")
    for _, row in vif_data.iterrows():
        flag = " ⚠️" if row['VIF'] > 5 else ""
        print(f"  {row['Variable']:25s} VIF = {row['VIF']:.2f}{flag}")
    return vif_data


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def resumen_dataset(df):
    """
    Muestra un resumen rápido del dataset: shape, tipos, nulos y estadísticas.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame a resumir.

    Ejemplo:
    --------
        resumen_dataset(df)
    """
    print(f"{'='*50}")
    print(f"Filas: {df.shape[0]:,}  |  Columnas: {df.shape[1]}")
    print(f"{'='*50}")

    print(f"\nValores nulos por columna:")
    nulos = df.isnull().sum()
    if nulos.sum() == 0:
        print("  Ninguno")
    else:
        for col in nulos[nulos > 0].index:
            print(f"  {col}: {nulos[col]} ({nulos[col]/len(df)*100:.1f}%)")

    print(f"\nEstadísticas descriptivas:")
    print(df.describe().to_string())


def listar_funciones():
    """
    Muestra todas las funciones disponibles en el toolbox con una breve descripción.

    Ejemplo:
    --------
        listar_funciones()
    """
    funciones = {
        "EXPLORACIÓN Y VISUALIZACIÓN": {
            "resumen_dataset(df)": "Resumen rápido: shape, nulos, describe",
            "matriz_correlacion(df, columnas, titulo)": "Heatmap de correlaciones",
            "scatter_pares(df, columnas, target)": "Pairplot entre variables",
            "histograma_distribucion(datos, titulo, bins)": "Histograma de una variable",
            "histogramas_multiples(df, columnas, bins)": "Grid de histogramas de varias variables",
            "boxplot_comparativo(df, columnas)": "Boxplots lado a lado",
            "distribucion_por_grupo(df, col_num, col_grupo)": "Boxplot/violin agrupado por categoría",
            "heatmap_valores_nulos(df)": "Mapa de calor de valores nulos",
        },
        "EVALUACIÓN — REGRESIÓN": {
            "scatter_real_vs_predicho(y_real, y_pred)": "Scatter de real vs predicho",
            "grafico_residuos(y_pred, residuos)": "Residuos vs predichos",
            "grafico_importancia_features(importances, names, top_n)": "Top features de un modelo",
            "grafico_coeficientes(coefs, names)": "Coeficientes de un modelo lineal",
            "tabla_metricas_regresion(y_real, y_pred, nombre)": "MSE, MAE, R² de un modelo",
            "comparar_modelos_regresion(modelos, X_tr, X_te, y_tr, y_te)": "Tabla comparativa de modelos",
        },
        "EVALUACIÓN — CLASIFICACIÓN": {
            "matriz_confusion(y_real, y_pred, etiquetas)": "Matriz de confusión + report",
            "grafico_probabilidades(probs, nombres)": "Barras de probabilidad por clase",
            "grafico_roc_auc(y_real, y_proba)": "Curva ROC + AUC",
            "grafico_precision_recall(y_real, y_proba)": "Curva Precision-Recall",
        },
        "DIAGNÓSTICO ESTADÍSTICO": {
            "test_shapiro_wilk(residuos)": "Test de normalidad (Shapiro-Wilk)",
            "test_kolmogorov_smirnov(datos)": "Test de normalidad (Kolmogorov-Smirnov)",
            "test_anderson_darling(datos)": "Test de normalidad (Anderson-Darling)",
            "test_breusch_pagan(residuos, X)": "Test de heterocedasticidad (Breusch-Pagan)",
            "test_white(residuos, X)": "Test de heterocedasticidad (White)",
            "test_levene(*grupos)": "Test de igualdad de varianzas (Levene)",
            "test_durbin_watson(residuos)": "Test de autocorrelación (Durbin-Watson)",
            "calcular_vif(df, columnas)": "Factor de inflación de varianza (multicolinealidad)",
            "qqplot(residuos)": "Gráfico Q-Q de normalidad",
            "curvas_aprendizaje(modelo, X, y)": "Curvas de aprendizaje (entrena el modelo)",
            "plot_curvas_precalculadas(sizes, train_sc, test_sc)": "Grafica curvas ya calculadas",
        },
    }

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║               TOOLBOX — Funciones disponibles               ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    for categoria, funcs in funciones.items():
        print(f"\n  📦 {categoria}")
        print(f"  {'─'*55}")
        for firma, desc in funcs.items():
            print(f"    • {firma}")
            print(f"      └─ {desc}")
    print()
