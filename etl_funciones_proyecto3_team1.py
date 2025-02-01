# ETL - FUNCIONES - PROYECTO 3 - TEAM 1 - OPTIMIZACIÓN DEL TALENTO

# =========================
# IMPORTACIÓN DE LIBRERÍAS
# =========================
# Tratamiento de datos:
import pandas as pd
import numpy as np
# Visualización:
import matplotlib.pyplot as plt
import seaborn as sns
# Evaluar linealidad de las relaciones entre las variables y la distribución de las variables:
import scipy.stats as st
import scipy.stats as stats
from scipy.stats import shapiro, poisson, chisquare, expon, kstest
# Librería para realizar el A/B Testing:
from statsmodels.stats.proportion import proportions_ztest
# Librerias para imputar nulos:
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# Importar librerías para conectar con MySQL:
import mysql.connector
from mysql.connector import errorcode
from sqlalchemy import create_engine  
import pymysql
# Importar librerías para transformar datos de categóricos a numéricos y viceversa:
from sklearn.preprocessing import LabelEncoder
# Configuración: Para poder visualizar todas las columnas de los DataFrames
pd.set_option('display.max_columns', None) 
# Gestión de los warnings:
import warnings
warnings.filterwarnings("ignore")

# ===========================================
# Configuración de la base de datos para SQL
# ===========================================
host = 'localhost'
user = 'root'
password = 'AdalabAlumnas'
database = 'bd_proyecto3'

# ===================================================== 
# EXTRACCIÓN: Obtener los datos desde los archivos CSV
# =====================================================
def extract_data(file_path):
    print("----------------------------------------------------------------")
    print(f"Extrayendo datos desde {file_path}...")
    df = pd.read_csv(file_path)
    print(df.head())
    print("----------------------------------------------------------------")
    return df

# ======================================
# EXPLORACIÓN: Exploración de los datos
# ======================================
def explore_data(df):
    print("----------------------------------------------------------------")
    print("Explorando los datos...\n")
    print('Aplicando ".info()"', df.info())
    print("----------------------------------------------------------------")
    print('Aplicando ".shape"', df.shape)
    print("----------------------------------------------------------------")
    print('Aplicando ".columns"', df.columns)
    print("----------------------------------------------------------------")
    print('Aplicando ".describe()"', df.describe())
    print("----------------------------------------------------------------")
    print('Aplicando ".isnull().sum()"', df.isnull().sum())
    print("----------------------------------------------------------------")
    print('Aplicando ".isnull().mean()*100).round(2).sort_values(ascending=False)"', (df.isnull().mean()*100).round(2).sort_values(ascending=False))
    print("----------------------------------------------------------------")
    print('Aplicando .duplicated().sum()"', df.duplicated().sum())
    print("----------------------------------------------------------------")
    return df

# ================================================
# TRANSFORMACIÓN: Limpiar y transformar los datos
# ================================================

# FUNCIÓN 1 - Transformar los registros a minúsculas, cuando proceda (strings con letras):
def transform_lower_rows(df):
    columns_object = df.select_dtypes(include=['object']).columns # Seleccionar las columnas str.
    df[columns_object] = df[columns_object].apply(lambda x: x.str.lower()) # aplicar la transformación a minúsculas.
    return df

# FUNCIÓN 2 - Transformar los nombres de las columnas:
def transform_column_name(column):
    # Dividir por mayúsculas o caracteres no alfabéticos y unir con "_":
    words = ''.join(['_' + c if c.isupper() and i != 0 else c for i, c in enumerate(column)]).split('_')
    words = [word.capitalize() for word in words if word]  # Poner en mayúscula la primera letra de cada palabra.
    return '_'.join(words)  # Unir con "_".

# FUNCIÓN 3 - Transformar variables categórica seleccionadas a numéricas:
def transform_numeric(df, col_categoric_numeric):
    for col in col_categoric_numeric: # iterar por las columnas que queremos transformar a numéricas.
        if col in df.columns: # verificar si la columna existe en el df.
            try:
                df[col] = df[col].astype(str).str.replace(',0$', '', regex=True) # eliminar ',0$'.
                df[col] = df[col].str.replace('$', '') # eliminar '$'.
                df[col] = df[col].str.replace("'", '') # quitar las comillas.
                df[col] = df[col].replace('Not Available', np.nan) # reemplazar 'Not Available' por un nulo 'nan'.
                df[col] = df[col].apply(lambda x: x.replace(',', '.') if isinstance(x, str) else x) # reemplar ',' por '.'
                #df[col] = pd.to_numeric(df[col], error='coerce') # convertir a float.
                df[col] = df[col].astype(float)
            except ValueError:
                # Manejar el error convirtiendo los valores no numéricos a NaN
                df[col] = df[col].apply(lambda x: pd.to_numeric(x, errors='coerce'))  #errors='coerce' para versiones antiguas
                #df[col] = df[col].astype(float) #Intenta convertir a float luego de haber manejado los errores
    return df

# FUNCIÓN 4 - Transformar variables numéricas seleccionadas a categóricas:
def transform_categoric(df, col_numeric_categoric):
    for col in col_numeric_categoric: # iterar por las columnas que queremos transformar a categóricas.
        if col in df.columns: # verificar si la columna existe en el df.
            df[col] = df[col].astype('object') # convertir a string.
    return df

# FUNCIÓN 5 - Reemplazar los números escritos a cifras:
def convertir_a_numero(valor):
    # Creamos un diccionario para mapear palabras a números:
    numeros_escritos = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 
                    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 
                    'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 
                    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
    if valor.isdigit(): # si ya es un número, devolverlo como int.
        return int(valor)
    else:
        try:
            partes = valor.split('-') # Dividimos el número escrito en decenas y unidades.
            if len(partes) == 2:
                decenas = numeros_escritos.get(partes[0].lower(), 0)
                unidades = numeros_escritos.get(partes[1].lower(), 0)
                return decenas + unidades # sumamos las cifras obtenidas.
            else:
                return numeros_escritos.get(valor.lower(), 0)
        except:
            return None

# FUNCIÓN 6 - Reemplazar valores negativos por nulos 'nan':
def transform_negative_values(df, column):
    df[column] = np.where(df[column] < 0, np.nan, df[column])
    return df

# FUNCIÓN 7 - Reemplazar valores con símbolos por espacios:
def transform_symbols(df, column):
    df[column] = df[column].str.replace(r"[_-]", " ", regex=True)
    return df

# FUNCIÓN 8 - Seleccionar el número de decimales de los float:
def transform_round_float(df, column, decimals):
    df[column] = df[column].round(decimals)
    return df

# FUNCIÓN PRINCIPAL DE TRANSFORMACIÓN DE DATOS:
def transform_data(df):
    print("----------------------------------------------------------------")
    print("Transformando datos...")

    # Casos únicos para la transformación del nombre de las columnas:
    df = df.rename(columns={'Unnamed: 0': 'IdEmployee', 'employeecount': 'EmployeeCount', 'employeenumber': 'EmployeeNumber',
                        'NUMCOMPANIESWORKED': 'NumCompaniesWorked', 'Over18': 'Over_18', 'TOTALWORKINGYEARS': 'TotalWorkingYears',
                        'WORKLIFEBALANCE': 'WorkLifeBalance', 'YEARSWITHCURRMANAGER': 'YearsWithCurrmanager', 'NUMBERCHILDREN': 'NumberChildren'})
    # Aplicar la transformación a los nombres de las columnas:
    df.columns = [transform_column_name(col) for col in df.columns]
    print('Transformando el nombre de las columnas para que tengan la misma escritura ', df.columns)
    print("----------------------------------------------------------------")

    # Transformar los registros a minúsculas, cuando proceda (strings con letras):
    df = transform_lower_rows(df)
    print('Transformando los registros a minúsculas, cuando proceda (string con letras)', df.head())
    print("----------------------------------------------------------------")

    # Transformar variables categóricas a numéricas:
    col_categoric_numeric = ['Daily_Rate', 'Hourly_Rate', 'Total_Working_Years', 'Monthly_Income', 'Same_As_Monthly_Income']
    df = transform_numeric(df, col_categoric_numeric)
    print('Transformando variables categóricas a numéricas', df[col_categoric_numeric].dtypes)
    print("----------------------------------------------------------------")

    # Transformar variables numéricas a categóricas:
    col_numeric_categoric = ['Job_Level', 'Education']
    df = transform_categoric(df, col_numeric_categoric)
    print('Transformando variables categóricas a numéricas', df[col_numeric_categoric].dtypes)
    print("----------------------------------------------------------------")

    # Reemplazar los valores negativos en nulos 'nan':
    column_negative_values = ['Distance_From_Home']
    df = transform_negative_values(df, column_negative_values)
    print("Transformando los valores negativos en nulos ", df[column_negative_values].min())
    print("----------------------------------------------------------------")

    # Transformar símbolos por espacios:
    col_symbols = 'Business_Travel'
    df = transform_symbols(df, col_symbols)
    df[col_symbols] = df[col_symbols].str.replace(r"[_-]", " ", regex=True)
    print("Transformando resgistros con símbolos por espacios ", df[col_symbols].head())
    print("----------------------------------------------------------------")

    # Seleccionar el número de decimales de los float:
    col_decimals_0 = ['Total_Working_Years']
    df = transform_round_float(df, col_decimals_0, 0)
    col_decimals_2 = ['Monthly_Income']
    df = transform_round_float(df, col_decimals_2, 2)

    # Arreglar columna 'Age': transformar números escritos en cifras:
    df['Age'] = df['Age'].apply(convertir_a_numero)
    print("Transformando todos los registros de 'Age' en cifras ", df['Age'].unique())
    print("----------------------------------------------------------------")

    # Arreglar columna 'Gender': transformar '0':'male' y '1':'female'
    df['Gender'] = df['Gender'].replace({0: 'male', 1: 'female'})
    print("Asignando género a los valores '0' y '1' ", df['Gender'].unique())
    print("----------------------------------------------------------------")

    # Arreglar columna 'Marital_Status': Corregir la escritura de 'marreid' por 'married':
    df['Marital_Status'] = df['Marital_Status'].replace({"marreid": 'married'})
    print("Corriendo la escritura de 'marreid' ", df['Marital_Status'].unique())
    print("----------------------------------------------------------------")

    # Arreglar la columna 'Remote_Work': unificar valores a 'yes' y 'no':
    # El criterio que seguimos es: ('Yes', '1', 'True'):'yes' y ('No', '0', 'False'):'no'.
    df['Remote_Work'] = df['Remote_Work'].replace({'Yes': 'yes', '1': 'yes', 'False': 'no', '0': 'no', 'True': 'yes'})
    print("Asignando 'yes' y 'no' a los diferentes valores ", df['Remote_Work'].unique())
    print("----------------------------------------------------------------")

    return df

# ==================================
# TRANSFORMACIÓN DE LOS DATOS NULOS
# ==================================

# FUNCIÓN 9 - Imputar la Moda en nulos (categóricas con bajo % nulos con categoría dominante):
def nulls_impute_mode(df, column):
    for col in column:
        moda = df[col].mode()[0]
        df[col] = df[col].fillna(moda)
    return df

# FUNCIÓN 10 - Imputar Nueva Categoría 'unknown' (categóricas con bajo % nulos sin categoría dominante):
def nulls_impute_newcategory(df, column):
    df[column] = df[column].fillna('unknown')
    return df

# FUNCIÓN 11 - Transformar datos categóricos a numéricos, imputar y reconvertir a categorías originales:
def transform_labelencoder(df, columns):
    le_dict = {}  # Diccionario para guardar los LabelEncoders de cada columna
    for column in columns:
        le = LabelEncoder()
        col_encoded = f"{column}_encoded"  # Nombre de la columna codificada.
        col_imputed = f"{col_encoded}_imputed"  # Nombre de la columna imputada.
        # Codificar la columna categórica:
        df[col_encoded] = le.fit_transform(df[column].fillna('missing'))
        le_dict[column] = le  # Guardamos el LabelEncoder para la reconversión.
    # Aplicar KNN Imputer:
    imputer = KNNImputer(n_neighbors=2)
    df[[f"{col}_encoded_imputed" for col in columns]] = imputer.fit_transform(df[[f"{col}_encoded" for col in columns]])
    # Reconvertir los valores imputados a las categorías originales:
    for column in columns:
        col_encoded = f"{column}_encoded"
        col_imputed = f"{col_encoded}_imputed"
        df[column] = le_dict[column].inverse_transform(df[col_imputed].round().astype(int))  # Convertir a categoría
    # Eliminar columnas residuales (_encoded y _imputed):
    df.drop(columns=[f"{col}_encoded" for col in columns] + [f"{col}_encoded_imputed" for col in columns], inplace=True)
    return df

# FUNCIÓN 12 - Aplicar las Técnicas Avanzadas de Imputación:
def nulls_knn_iterative_imputer(df, columns):
    imputer_iterative = IterativeImputer(max_iter=20, random_state=42)
    imputer_knn = KNNImputer(n_neighbors=2)
    for column in columns:
        # Aplicar Iterative Imputer:
        col_iterative = f"{column}_iterative"
        df[col_iterative] = imputer_iterative.fit_transform(df[[column]])
        # Aplicar KNN Imputer:
        col_knn = f"{column}_knn"
        df[col_knn] = imputer_knn.fit_transform(df[[column]])
        # Mostrar estadísticas de comparación:
        print(f"Comparación de imputación para {column}:\n")
        print(df[[column, col_iterative, col_knn]].describe().T)
        print("-" * 50)
    # Eliminar las columnas originales y las imputadas con Iterative Imputer:
    df.drop(columns=[col for column in columns for col in [column, f"{column}_iterative"]], inplace=True)
    # Renombrar las columnas KNN con su nombre original:
    df.rename(columns={f"{column}_knn": column for column in columns}, inplace=True)
    return df

# FUNCIÓN 13 - Imputar la Mediana:
def nulls_impute_median(df, column):
    for col in column:
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)
        return df
    
# FUNCIÓN PRINCIPAL DE TRANSFORMACIÓN DE NULOS:
def transform_nulls(df):
    # Listar los nulos por porcentaje de mayor a menor:
    nulos = (df.isnull().sum()/df.shape[0]*100).sort_values(ascending=False)
    nulos = nulos[nulos > 0]
    # Lo transformamos en un Data Frame:
    nulos = nulos.to_frame(name='perc_nulos').reset_index().rename(columns={'index': 'var'})
    print(nulos)
    print("--------------------------------------------------------------------")
    # SELECCIÓN DE VARIABLES CATEGÓRICAS CON NULOS:
    # 1. Vamos a seleccionar las categóricas con nulos, creando la intersección entre ambos criterios:
    columnas_object = df.select_dtypes(include=['object']).columns
    columnas_nulos = nulos['var'].to_list()
    columnas_object_nulos = columnas_object.intersection(columnas_nulos)
    # 2. Obtenemos la proporción de cada categoría en su propia columna:
    for col in columnas_object_nulos:
        print(f"La distribución de las categorías para la columna", col)
        print((df[col].value_counts() / df.shape[0]) * 100)  
        print("--------------------------------------------------------------------")
    print("--------------------------------------------------------------------")
    '''ESTRATEGIAS A SEGUIR CON VARIABLES CATEGÓRICAS Y CON NULOS:
    1. Alto % nulos (> 30%): Imputación con la Moda o con Técnicas Avanzadas:
        - 'Bussines_Travel' - 47.8% nulos: --> Técnicas Avanzadas.
        - 'Education_Field' - 46.1% nulos: --> Técnicas Avanzadas.
        - 'Marital Status' - 40.3% nulos: --> Técnicas Avanzadas.
        - 'Over Time' - 41.8% nulos: --> Técnicas Avanzadas.
    2. Bajo % nulos (< 30%): Imputación con la Moda (variable dominante) o con una Nueva Categoría (variable no dominate):
        - 'Employee Number' - 26.7% nulos --> Nueva Categoría 'Unknown'.
        - 'Performance Rating' - 12.0% nulos:
            - 3,0    74.659232 --> MODA.
            - 4,0    13.258984 
        - 'Work Life Balance' - 6.6% nulos:
            - 3,0    56.567534 --> MODA.
            - 2,0    22.242875
            - 4,0     9.603470
            - 1,0     4.894672'''
    # SELECCIÓN DE VARIABLES NUMÉRICAS CON NULOS:
    # 1. Vamos a seleccionar las numéricas con nulos, creando la intersección entre ambos criterios:
    columnas_number = df.select_dtypes(include=['number']).columns
    columnas_nulos = nulos['var'].to_list()
    columnas_number_nulos = columnas_number.intersection(columnas_nulos)
    # 2. Creamos un histograma, por separado, para cada columna seleccionada:
    for col in list(columnas_number_nulos):
        plt.figure(figsize=(8, 5))
        plt.hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black')
        plt.title(f'Histograma de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.show()
    print("--------------------------------------------------------------------")
    '''ESTRATEGIAS A SEGUIR CON VARIABLES NUMÉRICAS Y CON NULOS:
    1. Alto % nulos (> 30%): Imputación de Técnicas Avanzadas:
        - 'Monthly Income' - 52.2% nulos
        - 'Total Working Years' - 32.5% nulos
    2. Bajo % nulos (< 30%):
        - 'Daily Rate' - 7.6% nulos - Distribución NO simétrica --> MEDIANA.
        - 'Hourly Rate' - 5.2% nulos - Distribución NO simétrica --> MEDIANA.
        - 'Distance For Home' - 11.9% nulos - Distribución NO simétrica --> MEDIANA.'''
    # Imputar la Moda en nulos (categóricas con bajo % nulos con categoría dominante):
    column_mode = ['Performance_Rating', 'Work_Life_Balance']
    df = nulls_impute_mode(df, column_mode)
    # Imputar Nueva Categoría en nulos (categóricas con bajo % nulos con categoría dominante):
    column_newcategory = ['Employee_Number']
    df = nulls_impute_newcategory(df, column_newcategory)
    # Imputar Técnica Avanzada KNN Imputer en nulos (categóricas con alto % nulos):
    # Transformar datos de categóricos a numéricos y viceversa (LabelEncoder):
    col_labelencoder = ['Business_Travel', 'Education_Field', 'Marital_Status', 'Over_Time']
    df = transform_labelencoder(df, col_labelencoder)
    # Imputar Técnicas Avanzadas en nulos (numéricas con alto % nulos):
    # Aplicar las Técnicas Avanzadas de Imputación:
    col_iterative_knn = ['Monthly_Income', 'Total_Working_Years']
    df = nulls_knn_iterative_imputer(df, col_iterative_knn)
    # Imputar Mediana en nulos (numéricas con bajo % nulos):
    col_median = ['Hourly_Rate', 'Daily_Rate', 'Distance_From_Home']
    df = nulls_impute_median(df, col_median)
    print("--------------------------------------------------------------------")
    return df


# ========================
# ELIMINACIÓN DE COLUMNAS
# ========================

# FUNCIÓN PRINCIPAL PARA ELIMINAR COLUMNAS:
def remove_column(df, col_remove):
    df.drop(columns = col_remove, inplace = True)
    return df
    print("Eliminando las columnas seleccionadas ", df.columns)
    print("--------------------------------------------------------------------")


# =======================================================
# DISEÑO BBDD: Crear la base de datos y cargar los datos
# =======================================================
def create_db():
    # Conectar a MySQL usando pymysql:
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password)
    # Crear un cursor:
    cursor = connection.cursor()
    # Crear una base de datos si no existe:
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
    print("Base de Datos creada exitosamente en MySQL.")
    # Cerrar la conexión:
    connection.close()

def load_data(table_name, data):
    print(f"Cargando datos en la tabla {table_name}...")
    # Crear conexión a MySQL usando SQLAlchemy:
    engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}/{database}')
    # Volvemos a conectarnos con SQL para crear las tablas:
    cnx = mysql.connector.connect(user=user, password=password, host=host, database=database)
    mycursor = cnx.cursor()
    # Dividimos el DataFrame con las tablas y sus columnas correspondientes:
    datos_education_level = data[['Id_Employee', 'Education', 'Education_Field']] 
    datos_job_category = data[['Id_Employee', 'Job_Level', 'Job_Role', 'Stock_Option_Level', 'Performance_Rating']] 
    datos_salary = data[['Id_Employee', 'Daily_Rate', 'Hourly_Rate', 'Monthly_Income', 'Monthly_Rate', 'Percent_Salary_Hike']]
    datos_logistics = data[['Id_Employee', 'Business_Travel', 'Distance_From_Home', 'Over_Time', 'Remote_Work']]
    datos_personal = data[['Id_Employee', 'Age', 'Date_Birth', 'Gender', 'Marital_Status', 'Remote_Work',  'Relationship_Satisfaction', 'Work_Life_Balance']]
    datos_company_work = data[['Id_Employee', 'Attrition', 'Environment_Satisfaction', 'Job_Involvement', 'Job_Satisfaction', 'Num_Companies_Worked', 
                                'Over_Time', 'Performance_Rating', 'Relationship_Satisfaction', 'Total_Working_Years', 'Training_Times_Last_Year', 
                                'Years_At_Company', 'Years_Since_Last_Promotion']]
    # Insertar datos desde el DataFrame en MySQL:
    datos_education_level.to_sql('education_level', con=engine, if_exists='append', index=False)
    datos_job_category.to_sql('job_category', con=engine, if_exists='append', index=False)
    datos_salary.to_sql('salary', con=engine, if_exists='append', index=False)
    datos_logistics.to_sql('logistics', con=engine, if_exists='append', index=False)
    datos_personal.to_sql('personal', con=engine, if_exists='append', index=False)
    datos_company_work.to_sql('company_work', con=engine, if_exists='append', index=False)
    print("Datos insertados exitosamente en sus correspondientes tablas.")
    cnx.close()

# ============
# A/B TESTING
# ============

# FUNCIÓN 14 - Asignar grupo:
def asignar_grupo(satisfaccion):
    if satisfaccion >= 3:
        return 'A'
    else:
        return 'B'

# FUNCIÓN 15 - Calcular la tasa de rotación de cada grupo:
def calcular_tasa_rotacion(df_testing, grupo):
    grupo_df_testing = df_testing[df_testing['Grupo'] == grupo]
    empleados_que_dejaron = grupo_df_testing['Attrition'].apply(lambda x: 1 if x == 'yes' else 0).sum()
    total_empleados = len(grupo_df_testing)
    tasa_rotacion = (empleados_que_dejaron / total_empleados) * 100
    return tasa_rotacion

# FUNCIÓN PRINCIPAL:
def ab_testing(df):
    df_testing = df

    # 1. Extraccion muestra datos:
    # Grupo A (Control): Empleados con un nivel de satisfacción en el trabajo igual o superior a 3 en una escala de 1 a 5
    # Grupo B (Variante): Empleados con un nivel de satisfacción en el trabajo inferior a 3 en la misma escala.
    # Aplicar la función a la columna 'satisfaccion' y crear la columna 'grupo':
    df_testing['Grupo'] = df_testing['Job_Satisfaction'].apply(asignar_grupo)
    # Mostrar el DataFrame resultante:
    print(df_testing)
    # Calcular la tasa de rotación para los grupos A y B
    tasa_rotacion_A = calcular_tasa_rotacion(df_testing, 'A')
    tasa_rotacion_B = calcular_tasa_rotacion(df_testing, 'B')
    print(f"Tasa de rotación en el Grupo A: {tasa_rotacion_A}%")
    print(f"Tasa de rotación en el Grupo B: {tasa_rotacion_B}%")
    # Hipótesis nula (H₀): No hay diferencia significativa en la tasa de rotación de ambos grupos.
    # 𝐻0:𝜇1=𝜇2
    # Hipótesis alternativa (H₁): sí hay diferencia significativa.
    # Elegir el nivel de significancia (α): 0.05
    # Elección estadístico: comparar variable categórica tenemos -- Chi Cuadrado y porcentajes.
    # Prueba de proporciones (prueba z): Mejor si quieres comparar las tasas de rotación (proporciones de Yes) entre dos grupos.
    # Prueba de chi-cuadrado: Mejor si quieres saber si existe una relación entre las variables categóricas, como Group y Attrition.
    # Hacemos las muestras, clasificando a los empleado por grupo.
    df_testing['Group'] = df_testing['Job_Satisfaction'].apply(lambda x: 'A, Alta Satisfacion' if x >= 3 else 'B, Baja Satisfacion')
    # A es "Alta satisfación" y B es "Baja satisfación".
    grupos = df_testing.groupby('Group')['Attrition'].value_counts().reset_index(name = 'Total')
    print(grupos)
    grupos_con_rotacion = grupos[grupos['Attrition'] == 'yes']
    print(grupos_con_rotacion)
    # Calcular la proporción de 'Yes' en cada grupo (tasa de rotación):
    proporcion_A = df_testing[(df_testing['Group'] == 'A, Alta Satisfacion') & (df_testing['Attrition'] == 'yes')].shape[0] / df_testing[df_testing['Group'] == 'A, Alta Satisfacion'].shape[0]
    proporcion_B = df_testing[(df_testing['Group'] == 'B, Baja Satisfacion') & (df_testing['Attrition'] == 'yes')].shape[0] / df_testing[df_testing['Group'] == 'B, Baja Satisfacion'].shape[0]
    # Calcular la diferencia de proporciones (diferencia de medias):
    diferencia = proporcion_A - proporcion_B
    # Mostrar los resultados:
    print(f'Proporción de rotación en el grupo A: {proporcion_A:.4f}')
    print(f'Proporción de rotación en el grupo B: {proporcion_B:.4f}')
    print(f'Diferencia de proporciones (tasa de rotación): {diferencia:.4f}')
    # Si la diferencia de proporciones es positiva, significa que el grupo A tiene una tasa de rotación más alta que el grupo B.
    # Si la diferencia es negativa, significa que el grupo B tiene una tasa de rotación más alta.
    # Si la diferencia es cercana a 0, significa que las tasas de rotación entre los grupos son similares.
    # EN ESTE CASO: diferencia negativa por lo que el grupo B tiene tasa de rotacion más alta
    # Hacer prueba de proporciones (prueba Z) para ver si la diferencia es significativa
    proporcion_A
    proporcion_B
    rotacion_A = df_testing[(df_testing['Group'] == 'A, Alta Satisfacion') & (df_testing['Attrition'] == 'yes')].shape[0]
    rotacion_B = df_testing[(df_testing['Group'] == 'B, Baja Satisfacion') & (df_testing['Attrition'] == 'yes')].shape[0]
    # Tamaño de las muestras
    tamaño_muestra_A = df_testing[df_testing['Group'] == 'A, Alta Satisfacion'].shape[0]
    tamaño_muestra_B = df_testing[df_testing['Group'] == 'B, Baja Satisfacion'].shape[0]
    # Contadores de 'Rotacion' y tamaños de muestra
    counts = [rotacion_A, rotacion_B]
    nobs = [tamaño_muestra_A, tamaño_muestra_B]
    # Realizar la prueba Z para dos proporciones
    stat, p_value = proportions_ztest(counts, nobs)
    # Mostrar resultados
    print(f'Estadístico Z: {stat}')
    print(f'Valor p: {p_value}')
    print('Interpretación \n')
    print('Si el valor p es menor que 0.05, puedes rechazar la hipótesis nula (que dice que las proporciones son iguales) \n')
    print('y concluir que hay una diferencia significativa en las tasas de rotación entre los dos grupos. \n')
    print('CONCLUSION: DIFERENCIA SIGNIFICATIVA \n')
    print('El estadístico Z negativo simplemente indica que el grupo A tiene una tasa de rotación más baja que el grupo B, porque al definir diferencia como A - B,\n')
    print('el hecho de ser negativo indica que el grupo B tiene una tasa de rotación más alta y esta diferencia es significativa \n')
    return df

# =====================
# Proceso ETL completo
# =====================
def etl_process():

    # Extraer datos:
    file_path = "HR RAW DATA.csv"
    df = extract_data(file_path)

    # Explorar los datos:
    explore_data(df)

    # Transformar datos:
    df = transform_data(df)

    # Transformar nulos:
    df = transform_nulls(df)

    # Crear una copia del data frame antes de eliminar columnas:
    df2 = df.copy()

    # Eliminar columnas:
    '''EXPLICACIÓN DE POR QUÉ ELIMINAMOS CADA COLUMNA:
    - 'Over 18' - columna que no nos aporta información útil para el estudio de los datos.
    - 'Number Children' - no nos aporta información porque son todos nulos.
    - 'Employee Count' - tiene el mismo valor '1', por lo que no aporta información.
    - 'Salary' - el dato es incoherente con el salario mensual y además es el mismo para todos.
    - 'Role Department' - es una combinación de la columna 'Education' y 'Education Field'.
    - 'Standar Hours' - un 75% de nulos y el resto es el mismo valor '80', por lo que no nos aporto información.
    - 'Years In Current Role' - 98% de nulos y sin información relevante.
    - 'Department' - 81% de nulos y sin información relevante.
    - 'Same_As_Monthly_Income' - es una columna duplicada de 'Monthly_Income'.'''
    # Selección de las columnas:
    col_remove = ['Over_18', 'Number_Children', 'Employee_Count', 'Salary', 'Role_Departament', 
              'Standard_Hours', 'Years_In_Current_Role', 'Department', 'Same_As_Monthly_Income']
    df2 = remove_column(df2, col_remove)

    # Guardar archivo nuevo .csv
    df2.to_csv("df_final.csv", index=False)

    # Crear la base de datos en SQL:
    # Leer el archivo con la base de datos que vamos a trabajar:
    datos = pd.read_csv("df_final.csv")
    create_db()

    # Cargar los datos en la base de datos en SQL:
    tables_names = ['education_level', 'job_category', 'salary', 'logistics', 'personal', 'company_work']
    load_data(tables_names, datos)

    # Realizar A/B Testing a los datos:
    ab_testing(df2)
