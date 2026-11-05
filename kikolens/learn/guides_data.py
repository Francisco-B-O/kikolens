"""Guía de lectura de cada sección: qué mirar y de qué desconfiar, en inglés y español, y los términos del glosario que ayudan.
Las claves son los nombres de las secciones del panel (hay una prueba que comprueba que existen).
(qué mirar en, qué mirar es, ojo en, ojo es, términos)"""

GUIDES = {
    "Overview": (
        "Start with the size of the table, the types of the columns and the share of missing values. Columns with many missing values or a single repeated value rarely help.",
        "Empieza por el tamaño de la tabla, los tipos de las columnas y la parte de valores que faltan. Las columnas con muchos valores vacíos o un único valor repetido rara vez ayudan.",
        "A column that looks numeric may be an identifier or a code; averaging it means nothing.",
        "Una columna que parece numérica puede ser un identificador o un código; hacer su media no significa nada.",
        ["outlier", "skewness"]),
    "Quality and cleaning": (
        "The score summarises problems found: missing values, duplicates, odd types and suspicious values. Fix what the issues list says in order of severity and re-check.",
        "La puntuación resume los problemas encontrados: valores que faltan, duplicados, tipos raros y valores sospechosos. Arregla lo que dice la lista por orden de gravedad y vuelve a comprobar.",
        "Filling missing values or deleting rows changes the data your conclusions rest on: note what you did.",
        "Rellenar valores que faltan o borrar filas cambia los datos sobre los que se apoyan tus conclusiones: apunta qué hiciste.",
        ["outlier", "data_leakage"]),
    "Statistics": (
        "Read the spread (standard deviation, quartiles) as much as the centre, and compare the mean with the median: if they differ a lot, the data are skewed or have extreme values.",
        "Lee la dispersión (desviación típica, cuartiles) tanto como el centro, y compara la media con la mediana: si difieren mucho, los datos son asimétricos o tienen valores extremos.",
        "A correlation does not say that one thing causes the other, and it misses curved relationships.",
        "Una correlación no dice que una cosa cause la otra, y no ve las relaciones curvas.",
        ["mean_median", "correlation", "skewness"]),
    "Experiments": (
        "Look at the size of the difference and its range first, then at the p-value. The page picks the test that suits your data and says why.",
        "Mira primero el tamaño de la diferencia y su rango, y después el p-valor. La página elige el contraste que conviene a tus datos y dice por qué.",
        "A significant result with a tiny difference is not worth acting on; a non-significant one with a small sample proves nothing. If you test many things, trust the corrected p-values.",
        "Un resultado significativo con una diferencia diminuta no merece actuar; uno no significativo con muestra pequeña no prueba nada. Si pruebas muchas cosas, fíate de los p-valores corregidos.",
        ["p_value", "effect_size", "power", "multiple_testing", "ab_test"]),
    "Sequential tests": (
        "The evidence line may be checked as often as you like; stop only when it crosses the boundary. The planner tells you how long an experiment is likely to take.",
        "La línea de evidencia puede mirarse tantas veces como quieras; para solo cuando cruce el límite. El planificador dice cuánto suele tardar un experimento.",
        "Stopping an ordinary test as soon as it looks significant inflates false alarms; that is what this method avoids.",
        "Parar un contraste normal en cuanto parece significativo infla las falsas alarmas; eso es lo que este método evita.",
        ["sequential", "p_hacking"]),
    "Impact": (
        "Compare what happened with what was expected without the change. The range shows how wrong that expectation has been before; an effect inside that range may be nothing.",
        "Compara lo que pasó con lo esperable sin el cambio. El rango muestra cuánto se ha equivocado esa expectativa antes; un efecto dentro de ese rango puede no ser nada.",
        "Something else that changed at the same time (a season, a campaign) looks like an effect of your change.",
        "Algo más que cambió a la vez (una temporada, una campaña) parece un efecto de tu cambio.",
        ["diff_in_diff", "confounder"]),
    "Causal effect": (
        "Read the adjusted estimates next to the raw difference: if they disagree, the groups were not comparable. Check the balance table, the overlap chart and the E-value.",
        "Lee las estimaciones ajustadas junto a la diferencia bruta: si no coinciden, los grupos no eran comparables. Mira la tabla de equilibrio, el gráfico de solape y el valor E.",
        "Adjusting only removes the confounders you measured. A hidden one can still explain the result; the E-value tells you how strong it would have to be.",
        "Ajustar solo quita los factores de confusión que mediste. Uno oculto puede seguir explicando el resultado; el valor E dice cuán fuerte tendría que ser.",
        ["propensity", "confounder", "causation"]),
    "Time series": (
        "Check the trend and the repeating pattern first, then whether the forecast beats the simple baselines on data it did not see.",
        "Mira primero la tendencia y el patrón que se repite, y luego si la previsión supera a las líneas base simples con datos que no vio.",
        "A forecast that does not beat 'repeat last season' is not worth its complexity. Uncertainty grows with distance.",
        "Una previsión que no supera a «repetir la última temporada» no justifica su complejidad. La incertidumbre crece con la distancia.",
        ["baseline", "seasonality", "backtest", "prediction_interval"]),
    "Classical series": (
        "Stationarity first: read the two tests together. Then the autocorrelation chart, the decomposition and the model's check against repeating the last season.",
        "Primero la estacionariedad: lee los dos contrastes juntos. Después el gráfico de autocorrelación, la descomposición y la comprobación del modelo frente a repetir la última temporada.",
        "When the two stationarity tests disagree the answer is not clear. Residuals that are still autocorrelated mean the model misses something.",
        "Cuando los dos contrastes de estacionariedad discrepan la respuesta no está clara. Si los residuos siguen autocorrelacionados, el modelo se deja algo.",
        ["stationarity", "autocorrelation", "arima", "seasonality"]),
    "Regression": (
        "Read each coefficient as 'the change in the outcome for one more unit, other things equal', next to its range. Check the warnings about unreliable errors.",
        "Lee cada coeficiente como «el cambio en el resultado por una unidad más, lo demás igual», junto a su rango. Atiende a los avisos de errores poco fiables.",
        "A high R squared does not mean the model is right, and predictors that overlap make individual coefficients unstable.",
        "Un R cuadrado alto no significa que el modelo sea correcto, y los predictores que se solapan hacen inestables los coeficientes.",
        ["regression_coef", "r_squared", "multicollinearity"]),
    "Model lab": (
        "Compare the models by their cross-validated score, not by how they fit the training data, and check that the best one clearly beats the baseline.",
        "Compara los modelos por su puntuación en validación cruzada, no por cómo ajustan los datos de entrenamiento, y comprueba que el mejor supera claramente a la línea base.",
        "A score that looks too good usually means data leakage. A gap between training and test scores means overfitting.",
        "Una puntuación demasiado buena suele significar fuga de datos. Una diferencia entre entrenamiento y prueba significa sobreajuste.",
        ["cross_validation", "overfitting", "data_leakage", "baseline", "roc_auc"]),
    "Models": (
        "Read the performance next to the baseline, then the features that matter and the plain-language reasons the page gives for its choices.",
        "Lee el rendimiento junto a la línea base, y después las variables que importan y las razones en lenguaje llano que da la página.",
        "Importance describes the model, not the world: changing an important input need not change the outcome.",
        "La importancia describe el modelo, no el mundo: cambiar una entrada importante no tiene por qué cambiar el resultado.",
        ["feature_importance", "baseline", "overfitting"]),
    "Segments": (
        "Check that the groups are stable and meaningful: sizes, how different they are and what the descriptions say.",
        "Comprueba que los grupos son estables y tienen sentido: tamaños, cuánto se diferencian y lo que dicen las descripciones.",
        "Clustering always finds groups, even in noise. A weak silhouette means weak structure.",
        "El agrupamiento siempre encuentra grupos, incluso en ruido. Una silueta débil significa poca estructura.",
        ["clustering", "silhouette", "pca"]),
    "Anomalies": (
        "Each flagged row comes with the reason it stands out. Look at the reasons before deciding whether it is an error or a real case.",
        "Cada fila marcada trae la razón por la que destaca. Mira las razones antes de decidir si es un error o un caso real.",
        "Unusual is not wrong: the most interesting cases are often the unusual ones.",
        "Inusual no es erróneo: los casos más interesantes suelen ser los inusuales.",
        ["outlier", "robust"]),
    "Survival": (
        "Read the curve as 'the share still going at each moment' and compare groups by how far apart the curves are and whether the test says the gap is more than chance.",
        "Lee la curva como «la parte que sigue en cada momento» y compara grupos por la distancia entre curvas y si el contraste dice que la diferencia es más que azar.",
        "Cases that have not had the event yet are information, not missing data.",
        "Los casos que aún no han tenido el suceso son información, no datos que faltan.",
        ["survival", "censoring", "hazard"]),
    "Text": (
        "Terms and topics show what people talk about; compare groups to see what differs. Topics are chosen by coherence, but look at the example texts to name them.",
        "Los términos y temas muestran de qué habla la gente; compara grupos para ver qué difiere. Los temas se eligen por coherencia, pero mira los textos de ejemplo para nombrarlos.",
        "Lexicon-based tone is approximate, and patterns such as emails are matched by format, not understood.",
        "El tono por léxico es aproximado, y los patrones como los correos se detectan por formato, no se entienden.",
        ["clustering"]),
    "Network": (
        "Look at who is central (PageRank, connections), which communities exist and how connected the whole network is. Bridges with few links can matter more than hubs.",
        "Mira quién es central (PageRank, conexiones), qué comunidades hay y lo conectada que está toda la red. Los puentes con pocos enlaces pueden importar más que los nodos centrales.",
        "Weak modularity means the groups found are not clearly separated.",
        "Una modularidad débil significa que los grupos encontrados no están claramente separados.",
        ["pagerank", "modularity"]),
    "Geography": (
        "Check the coordinates first, then distances and zones. For a pattern in space, read Moran's I and the hot and cold spots after correction.",
        "Revisa primero las coordenadas, luego distancias y zonas. Para un patrón en el espacio, lee la I de Moran y los focos calientes y fríos tras la corrección.",
        "Neighbouring places are not independent, which breaks ordinary tests. Hot spots found by looking at a map are leads, not proofs.",
        "Los lugares vecinos no son independientes, lo que rompe los contrastes normales. Los focos que se ven en un mapa son pistas, no pruebas.",
        ["moran", "multiple_testing"]),
    "Survey": (
        "For scale questions read the share favourable as well as the average. Alpha says whether questions measure the same thing; key drivers say what matters most for the overall rating.",
        "En las preguntas de escala lee el porcentaje favorable además de la media. El alfa dice si las preguntas miden lo mismo; los factores clave, qué pesa más en la valoración global.",
        "Scale steps are ordered but not equally spaced. Importance from survey data shows association, not cause.",
        "Los pasos de una escala están ordenados pero no equiespaciados. La importancia en datos de encuesta muestra asociación, no causa.",
        ["likert", "cronbach", "nps"]),
    "Why it changed": (
        "The table lists which values of which columns account for the change between two periods. Start with the biggest contributors.",
        "La tabla enumera qué valores de qué columnas explican el cambio entre dos periodos. Empieza por los que más contribuyen.",
        "It explains where the change came from, not why it happened.",
        "Explica de dónde vino el cambio, no por qué ocurrió.",
        ["causation"]),
    "Feature selection": (
        "Columns kept by several methods at once are the safest. The curve shows how many are enough.",
        "Las columnas que eligen varios métodos a la vez son las más seguras. La curva muestra cuántas bastan.",
        "Selecting columns with the same data you evaluate on inflates the score.",
        "Elegir columnas con los mismos datos con los que evalúas infla la puntuación.",
        ["cross_validation", "data_leakage"]),
    "Forecast with drivers": (
        "See whether adding the other columns really beats forecasting the series alone, and try the what-if scenarios.",
        "Mira si añadir las otras columnas supera de verdad a prever la serie sola, y prueba los escenarios.",
        "Predicting well does not mean that changing a driver would move the outcome.",
        "Predecir bien no significa que cambiar una de las variables mueva el resultado.",
        ["causation", "baseline"]),
    "Model governance": (
        "The stage shows how far a model has gone. Read the checks that failed and the validation range before moving it on.",
        "La etapa muestra hasta dónde ha llegado un modelo. Lee las comprobaciones que fallan y el rango de la validación antes de promoverlo.",
        "The log records who did what as typed; it is a record for the team, not access control.",
        "El registro anota quién hizo qué tal como se escribe; es un registro para el equipo, no control de acceso.",
        ["cross_validation", "baseline"]),
}
