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
        self.validate_data()

    def read_file(self, file_path: str) -> pd.DataFrame:
        """Читает файл с определением разделителя (CSV или TSV).

        Аргументы:
            file_path (str): Путь к файлу.

        Возвращает:
            pd.DataFrame: DataFrame с данными из файла.

        Исключения:
            pd.errors.ParserError: Если файл не может быть прочитан как CSV или TSV.
        """
        return pd.read_csv(file_path, sep=";", encoding="utf-8-sig")
        # try:
        #     return pd.read_csv(file_path, sep=";", encoding='utf-8-sig')  # Читаем как CSV
        # except pd.errors.ParserError:
        #     return pd.read_csv(file_path, sep="\t", encoding='utf-8-sig')  # Если ошибка, читаем как TSV

    def validate_data(self) -> None:
        """Проверяет совместимость данных в true_csv и pred_csv.

        Исключения:
            ValueError: Если количество строк не совпадают.
        """
        if len(self.true_csv) != len(self.pred_csv):
            raise ValueError("Количество строк true_csv и pred_csv не совпадает.")

    def calculate_metrics_by_id(self) -> pd.DataFrame:
        """Вычисляет метрики для каждого ID.

        Возвращает:
            pd.DataFrame: DataFrame с метриками для каждого ID.

        Метрики:
            - WER (Word Error Rate)
            - CER (Character Error Rate)
            - BLEU (Bilingual Evaluation Understudy)
        """
        merged_df = pd.merge(self.pred_csv, self.true_csv, left_on="id", right_on="id")

        # Функция для вычисления метрик
        def calculate_metrics(row):
            try:
                Y_true = ast.literal_eval(row["answer"])
            except:
                Y_true = row["answer"]

            try:
                y_pred = ast.literal_eval(row["answer"])
            except:
                y_pred = row["answer"]

            # Вычисляем метрики
            wer_error = wer(Y_true, y_pred)
            cer_error = cer(Y_true, y_pred)
            bleu_score = sacrebleu.corpus_bleu(Y_true, [y_pred]).score

            return pd.Series(
                {
                    "wer_error": wer_error,
                    "cer_error": cer_error,
                    "bleu_score": bleu_score,
                }
            )

        # Применяем функцию calculate_metrics к каждой строке
        metrics_df = merged_df.apply(calculate_metrics, axis=1)

        # Объединяем результаты с исходным DataFrame
        result_df = pd.concat([merged_df, metrics_df], axis=1)

        return result_df

    def calculate_metrics_by_doc_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Вычисляет метрики для каждого типа документа.

        Аргументы:
            df (pd.DataFrame): DataFrame из метода calculate_metrics_by_id.

        Возвращает:
            pd.DataFrame: DataFrame с метриками для каждого типа документа.
        """
        results = []

        # Группируем данные по doc_class
        grouped = df.groupby("doc_class")

        for doc_class, group in grouped:
            true_answers = group["answer"].tolist()
            pred_answers = group["model_answer"].tolist()

            # Вычисляем метрики
            wer_error = wer(true_answers, pred_answers)
            cer_error = cer(true_answers, pred_answers)
            bleu_error = sacrebleu.corpus_bleu(true_answers, [pred_answers]).score

            # Добавляем результат в список
            results.append(
                {
                    "doc_class": doc_class,
                    "wer_error": wer_error,
                    "cer_error": cer_error,
                    "bleu_error": bleu_error,
                }
            )

        # Создаем DataFrame из списка результатов
        result_df = pd.DataFrame(results)
        return result_df

    def group_by_doc_question(self, df: pd.DataFrame) -> pd.DataFrame:
        """Группирует данные по типу документа и типу вопроса.

        Аргументы:
            df (pd.DataFrame): Исходный DataFrame.

        Возвращает:
            pd.DataFrame: Сгруппированный DataFrame с метриками.
        """
        # Список для хранения результатов
        results = []

        # Группируем данные по doc_class и question_type
        grouped = df.groupby(["doc_class", "question_type"])

        for (doc_class, question_type), group in grouped:
            true_answers = group["answer"].tolist()
            pred_answers = group["pred_answers"].tolist()

            # Вычисляем метрики
            wer_error = wer(true_answers, pred_answers)
            cer_error = cer(true_answers, pred_answers)
            bleu_error = sacrebleu.corpus_bleu(true_answers, [pred_answers]).score

            # Добавляем результат в список
            results.append(
                {
                    "doc_class": doc_class,
                    "question_type": question_type,
                    "wer_error": wer_error,
                    "cer_error": cer_error,
                    "bleu_error": bleu_error,
                }
            )

        # Создаем DataFrame из списка результатов
        result_df = pd.DataFrame(results)
        return result_df

    def calculate_metrics_general(self) -> pd.DataFrame:
        """Вычисляет общие метрики по всему корпусу данных.

        Возвращает:
            dict: Словарь с метриками WER, CER и BLEU.
        """

        def safe_literal_eval(value):
            try:
                return ast.literal_eval(value)
            except:
                return value

        true_answers = self.true_csv["answer"].map(safe_literal_eval).explode().tolist()
        pred_answers = (
            self.pred_csv["model_answer"].map(safe_literal_eval).explode().tolist()
        )

        wer_error = wer(true_answers, pred_answers)
        cer_error = cer(true_answers, pred_answers)
        bleu_score = sacrebleu.corpus_bleu(true_answers, [pred_answers]).score

        result = {
            "wer_error": [wer_error],
            "cer_error": [cer_error],
            "bleu_score": [bleu_score],
        }

        return pd.DataFrame(result)

    def save_function_results(
        self, csv_path, func_name, func_arg=None, encoding="utf-8-sig", index=False
    ):
        """
        Описание ...
        """
        result = func_name(func_arg) if func_arg else func_name()
        result.to_csv(
            csv_path=csv_path,
            sep=";",
            encoding=encoding,
            index=index,
        )
        print(f"Результат метода {func_name.__name__} сохранен в папку {csv_path}")
