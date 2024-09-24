from pyspark import SparkContext, SparkConf, HiveContext

from pyspark.sql.functions import col, datediff, when, count, isnan, to_date, max as sparkmax
import pyspark.sql.types as T

from pyspark.sql import SparkSession, DataFrame

from pyspark.ml.feature import StringIndexer, OneHotEncoder

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

from colorama import Fore
import numpy as np
import matplotlib.pyplot as plt


def get_spark_conf(app_name: str, **kwargs):
    """Установка и получение кофигурации Spark"""
    
    BASE_CONFS = [
        ('spark.dynamicAllocation.enabled', 'True'),
        ('spark.executor.memory', '48g'),
        ('spark.driver.memory', '48g'),
        ('spark.yarn.executor.memoryOverhead', '36g'),
        ('spark.driver.maxResultSize', '128g'),
        ('spark.executor.cores', '15'),    
        ('spark.driver.maxResultSize','512g'),
        ('spark.dynamicAllocation.initialExecutors', 1),
        ('spark.dynamicAllocation.minExecutors', 1),
        ('spark.dynamicAllocation.maxExecutors', 5),
        ('spark.port.maxRetries', '150'),
        ('spark.driver.cores', '1'),
        ('spark.yarn.driver.memoryOverhead', '128g'),
        ('spark.shuffle.service.enabled', 'true'),
        ('spark.sql.legacy.parquet.datetimeRebaseModeInRead', 'CORRECTED'),
        ('spark.kryoserializer.buffer.maxValue', '2044018'),
        ('spark.sql.autoBroadcastJoinThreshold', -1),
        ('spark.sql.broadcastTimeout', -1),
        ("spark.sql.catalogImplementation", "hive"),
        ('spark.sql.parquet.datetimeRebaseModeInRead', 'CORRECTED'),
        ('spark.sql.parquet.int96RebaseModeInWrite', 'CORRECTED'),
        ('spark.sql.parquet.int96RebaseModeInRead', 'CORRECTED'),
    ]
    
    assert type(app_name) is str
    master = kwargs.pop('master', 'yarn')
    assert type(master) is str
    confs = kwargs.pop('conf', BASE_CONFS)
    assert type(confs) in (list, tuple)
    assert all(lambda c: type(c) in (list, tuple) for c in confs)
    assert all(lambda c: len(c) == 2 for c in confs)
    
    conf = SparkConf().setAppName(app_name).setMaster(master)
    for c in confs: conf.set(*c)
    
    return conf


def get_spark_session(app_name: str, **kwargs):
    """Получение или создание сессии Spark"""
    
    assert type(app_name) is str
    
    conf = get_spark_conf(app_name, **kwargs)
    
    print('При протухшем тикете: "!kinit" в терминале Юпитера и перезапуск ядра')
    
    return SparkSession.builder.appName(app_name).config(conf=conf).getOrCreate()

def shape(df) -> tuple:
    """Количественный размер DataFrame"""
    return df.count(), len(df.columns)

def count_na(df, columns=None, show=True) -> dict:
    """Подсчет и отображение количества пропусков"""
    
    columns = df.columns if columns is None else columns
    assert type(columns) in (tuple, list)
    assert all(lambda c: c in columns for c in df.columns)
    
    result = {columns[i]: cnt 
              for i, cnt in enumerate(df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) 
                                              for c in columns]).collect()[0])}
    
    if show:
        for key, value in result.items():
            print(f'{key}: {value}')
    
    return result

def summary(df, n=20, show=True, vertical=False, truncate=10) -> dict:
    """Вывод краткой информации о DataFrame"""
    result = dict()
    result['shape'] = shape(df)
    result['type'] = {c: dtype for c, dtype in df.dtypes}
    result['na'] = count_na(df, show=False)
    
    if show:
        print(f'shape: {result["shape"]}')
        print('columns')
        for c in df.columns:
            print('\t' + f'{c}' + Fore.YELLOW + f' ({result["type"][c]}) ' + Fore.RED + f'{result["na"][c]}' + Fore.RESET)
        print()
        df.show(n, vertical=vertical, truncate=truncate)
    
    return result

def unique(df, columns=None, show=True) -> dict:
    """Определение уникальных значений и их количества"""
    
    columns = df.columns if columns is None else columns
    assert type(columns) in (tuple, list)
    assert all(lambda c: c in columns for c in df.columns)
    
    result = dict()
    for column in columns:
        result[column] = {row[0]: row[1] for row in df.groupby(column).count().collect()}
    
    if show: 
        for key, value in result.items():
            print(f'{key}:')
            for k, v in value.items():
                print('\t' + f'{k}: {v}')
            
    return result

def vectorization(df, columns: list):
    """Векторизация признаков"""
    va = VectorAssembler(inputCols=columns, outputCol='features')
    return va.transform(df)

def encode_label(df, inputCol:str, outputCol:str, drop=False):
    """Преобразование n категорий в числа от 1 до n"""    
    le = StringIndexer(inputCol=inputCol, outputCol=outputCol)
    le = le.fit(df)
    df = le.transform(df)
    df = df.withColumn(outputCol, col(outputCol).cast("int"))
    if drop: df = df.drop(inputCol)
    return df

def encode_one_hot(df, inputCol:str, outputCol:str, drop=False):
    """Преобразование n значений каждой категории в n бинарных категорий"""    
    df = encode_label(df, inputCol, inputCol + '_label', drop=False)
    ohe = OneHotEncoder(inputCol=inputCol + '_label', outputCol=outputCol)
    ohe = ohe.fit(df)
    df = ohe.transform(df)
    if drop: df = df.drop(inputCol)
    return df

def show_corr(df, show_num=True, fmt=4, **kwargs):
    """Построение матрицы корреляци"""
    
    assert type(fmt) is int
    assert 1 <= fmt
    
    num_columns = [c for c, dtype in df.dtypes if dtype in ["int", "float", "double"]]
    df = df.select(num_columns)
    
    for column in df.columns: df = df.na.fill({column: 0}) 
    
    df = vectorization(df, df.columns)
    
    cor = Correlation.corr(df, 'features', method='pearson').collect()[0][0].toArray()
    cor = cor[:len(df.columns), :len(df.columns)]
    cor = cor.round(fmt)
    
    df = df.drop('features')
    
    fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (12, 12)))
    plt.suptitle('Correlation', fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    im = ax.imshow(cor, interpolation='nearest', cmap=kwargs.pop('cmap', 'bwr'))
    
    clrbar = fig.colorbar(im, orientation='vertical')
    clrbar.ax.tick_params(labelsize=12)
    clrbar.set_label('Color intensity []', fontsize=14)
    clrbar.set_ticks(np.linspace(-1, 1, 21))
    
    ax.set_xticks(range(len(df.columns)), df.columns, fontsize=12, rotation=90)
    ax.set_yticks(range(len(df.columns)), df.columns, fontsize=12, rotation=0)
    ax.tick_params(top=False, bottom=False, labeltop=False, labelbottom=True)
    
    if show_num:
        for row in range(len(df.columns)):
            for col in range(len(df.columns)):
                ax.text(row, col, cor[row, col], ha='center', va='center', color='black', fontsize=12, rotation=45)
    
    plt.show()