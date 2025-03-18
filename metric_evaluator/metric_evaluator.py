import pandas as pd
from jiwer import wer, cer
import sacrebleu
import ast


class MetricEvaluator:
    """Класс для вычисления метрик качества предсказаний модели.

    Атрибуты:
        true_csv (pd.DataFrame): DataFrame с правильными ответами.
        pred_csv (pd.DataFrame): DataFrame с предсказаниями модели.
    """

    def __init__(self, true_file: str, prediction_file: str) -> None:
        """Инициализирует экземпляр MetricEvaluator.

        Аргументы:
            true_file (str): Путь к файлу с правильными ответами (CSV или TSV).
            prediction_file (str): Путь к файлу с предсказаниями модели (CSV или TSV).

        Исключения:
            ValueError: Если данные в файлах не совместимы.
        """
        self.true_csv = self.read_file(true_file)
        self.true_csv["id"] = self.true_csv.index
        self.pred_csv = self.read_file(prediction_file)
        self.by_id_cache = None
        
        # поддерживаемые метрики и агрегаторы
        self.supported_metrics = {"WER", "CER", "BLEU"}

        # Фильтрация каждого DataFrame
        self.true_csv = self.true_csv[self.true_csv["id"].isin(self.pred_csv["id"])]
        self.pred_csv = self.pred_csv[self.pred_csv["id"].isin(self.true_csv["id"])]

        # Преобразуем значения в строки
        self.true_csv["answer"] = self.true_csv["answer"].astype(str)
        self.pred_csv["model_answer"] = self.pred_csv["model_answer"].astype(str)

        self.validate_data()
        
    def clear_by_id_cache(self):
        """Очистить кэш для вычисленного by_id агрегатора.
        """
        self.by_id_cache = None

    def read_file(self, file_path: str) -> pd.DataFrame:
        """Читает файл с определением разделителя (CSV или TSV).

        Аргументы:
            file_path (str): Путь к файлу.

        Возвращает:
            pd.DataFrame: DataFrame с данными из файла.
        """
        return pd.read_csv(file_path, sep=";", encoding="utf-8-sig")

    def validate_data(self) -> None:
        """Проверяет совместимость данных в true_csv и pred_csv."""
        if len(self.true_csv) != len(self.pred_csv):
            raise ValueError("Количество строк true_csv и pred_csv не совпадает.")

    def calculate_metrics_by_id(self, metrics: list = None) -> pd.DataFrame:
        """Вычисляет метрики для каждого ID.

        Аргументы:
            metrics (list): Список метрик для расчета (поддерживаются "WER", "CER", "BLEU").

        Возвращает:
            pd.DataFrame: DataFrame с метриками для каждого ID.
        """
        metrics = [m.upper() for m in metrics] if metrics else ["WER", "CER", "BLEU"]
        
        merged_df = pd.merge(
            self.pred_csv,
            self.true_csv,
            on="id",
            how="inner",
        )

        def calculate_metrics(row):
            Y_true = row["answer"]
            y_pred = row["model_answer"]

            metrics_dict = {}
            if "WER" in metrics:
                metrics_dict["wer_error"] = wer(Y_true, y_pred)
            if "CER" in metrics:
                metrics_dict["cer_error"] = cer(Y_true, y_pred)
            if "BLEU" in metrics:
                metrics_dict["bleu_score"] = sacrebleu.corpus_bleu([Y_true], [[y_pred]]).score

            return pd.Series(metrics_dict)

        metrics_df = merged_df.apply(calculate_metrics, axis=1)
        result_df = pd.concat([merged_df, metrics_df], axis=1)
        return result_df

    def calculate_metrics_by_doc_type(self, metrics: list = None) -> pd.DataFrame:
        """Вычисляет метрики для каждого типа документа.

        Аргументы:
            metrics (list): Список метрик для расчета.

        Возвращает:
            pd.DataFrame: DataFrame с метриками для каждого типа документа.
        """
        if self.by_id_cache is None:
            self.by_id_cache = self.calculate_metrics_by_id()
        
        metrics = [m.upper() for m in metrics] if metrics else ["WER", "CER", "BLEU"]
        results = []
        grouped = self.by_id_cache.groupby("doc_class")

        for doc_class, group in grouped:
            true_answers = group["answer"].tolist()
            pred_answers = group["model_answer"].tolist()

            metrics_dict = {"doc_class": doc_class}
            if "WER" in metrics:
                metrics_dict["wer_error"] = wer(true_answers, pred_answers)
            if "CER" in metrics:
                metrics_dict["cer_error"] = cer(true_answers, pred_answers)
            if "BLEU" in metrics:
                metrics_dict["bleu_error"] = sacrebleu.corpus_bleu(true_answers, [pred_answers]).score

            results.append(metrics_dict)

        return pd.DataFrame(results)

    def calculate_metrics_by_doc_question(self, metrics: list = None) -> pd.DataFrame:
        """Группирует данные по типу документа и вопроса.

        Аргументы:
            metrics (list): Список метрик для расчета.

        Возвращает:
            pd.DataFrame: Сгруппированный DataFrame с метриками.
        """
        if self.by_id_cache is None:
            self.by_id_cache = self.calculate_metrics_by_id()
        
        metrics = [m.upper() for m in metrics] if metrics else ["WER", "CER", "BLEU"]
        results = []
        grouped = self.by_id_cache.groupby(["doc_class", "question_type"])

        for (doc_class, question_type), group in grouped:
            true_answers = group["answer"].tolist()
            pred_answers = group["model_answer"].tolist()

            metrics_dict = {
                "doc_class": doc_class,
                "question_type": question_type,
            }
            if "WER" in metrics:
                metrics_dict["wer_error"] = wer(true_answers, pred_answers)
            if "CER" in metrics:
                metrics_dict["cer_error"] = cer(true_answers, pred_answers)
            if "BLEU" in metrics:
                metrics_dict["bleu_error"] = sacrebleu.corpus_bleu(true_answers, [pred_answers]).score

            results.append(metrics_dict)

        return pd.DataFrame(results)

    def calculate_metrics_general(self, metrics: list = None) -> pd.DataFrame:
        """Вычисляет общие метрики по всему корпусу данных.

        Аргументы:
            metrics (list): Список метрик для расчета.

        Возвращает:
            pd.DataFrame: DataFrame с общими метриками.
        """
        metrics = [m.upper() for m in metrics] if metrics else ["WER", "CER", "BLEU"]
        
        true_answers = self.true_csv["answer"].tolist()
        pred_answers = self.pred_csv["model_answer"].tolist()

        result = {}
        if "WER" in metrics:
            result["wer_error"] = [wer(true_answers, pred_answers)]
        if "CER" in metrics:
            result["cer_error"] = [cer(true_answers, pred_answers)]
        if "BLEU" in metrics:
            result["bleu_score"] = [sacrebleu.corpus_bleu(true_answers, [pred_answers]).score]

        return pd.DataFrame(result)

    def save_function_results(
        self, 
        csv_path, 
        func_name,
        metrics: list = None,
        encoding="utf-8-sig", 
        index=False
    ):
        """
        Сохраняет результаты работы методов класса в CSV-файл.

        Аргументы:
            csv_path (str): Путь для сохранения файла
            func_name (str): Имя метода ("by_id", "by_doc_type", "by_doc_question", "general")
            metrics (list): Список метрик для расчета
            encoding (str): Кодировка файла
            index (bool): Сохранять индекс DataFrame
        """
        func_map = {
            "by_id": self.calculate_metrics_by_id,
            "by_doc_type": self.calculate_metrics_by_doc_type,
            "by_doc_question": self.calculate_metrics_by_doc_question,
            "general": self.calculate_metrics_general,
        }

        if func_name in func_map:
            metric_func = func_map[func_name]
            result = metric_func(metrics=metrics)
            result.to_csv(csv_path, sep=";", encoding=encoding, index=index)
            print(f"Результат метода {metric_func.__name__} сохранен в {csv_path}")
            return result
        else:
            print(f"Указный вами метод агрегации {func_name} не поддерживается")
            supported_aggregators = ", ".join(tuple(func_map.keys()))
            print(f"Поддерживаются методы: {supported_aggregators}")

