"""Build a readable experiment summary from metrics.json and saved figures."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def main() -> None:
    data = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    labels = {"dummy": "Частый класс", "logistic": "Логистическая регрессия",
              "forest": "Случайный лес"}
    rows = "".join(
        f"<tr><td>{labels[name]}</td><td>{domain}</td>"
        f"<td>{score['balanced_accuracy']:.3f}</td><td>{score['f1']:.3f}</td>"
        f"<td>{score['roc_auc']:.3f}</td></tr>"
        for name, results in data["models"].items()
        for domain, score in results.items()
    )
    logistic_high = data["models"]["logistic"]["high"]["balanced_accuracy"]
    logistic_low = data["models"]["logistic"]["low"]["balanced_accuracy"]
    page = f"""<!doctype html><html lang='ru'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Рамановские спектры — наличие глюкозы</title><style>
:root{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#1c3440;background:#f4f7f7}}
*{{box-sizing:border-box}}body{{margin:0}}header{{background:#183e49;color:white;padding:42px 20px}}
header>div,main{{max-width:1080px;margin:auto}}main{{padding:26px 20px 60px}}
h1{{font-size:clamp(29px,4vw,42px);margin:6px 0}}h2{{font-size:22px;margin:0 0 16px}}
p{{line-height:1.6}}.eyebrow{{color:#9edacf;font-size:12px;letter-spacing:.11em;text-transform:uppercase}}
.sub{{color:#d0e0e1}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:16px}}
.card,section{{background:white;border:1px solid #dbe7e7;border-radius:12px;padding:21px}}
.card span{{font-size:13px;color:#5c747d}}.card strong{{display:block;font-size:28px;margin:7px 0}}
section{{margin-bottom:16px}}.split{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
img{{max-width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{border-bottom:1px solid #e5eeee;padding:10px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{color:#5c747d;font-weight:600}}.scroll{{overflow:auto}}.note{{color:#5c747d;font-size:14px}}
a{{color:#146e7a}}@media(max-width:700px){{.grid,.split{{grid-template-columns:1fr}}}}
</style></head><body><header><div><div class='eyebrow'>Рамановская спектроскопия · воспроизводимый эксперимент</div>
<h1>Есть ли глюкоза в смеси четырёх сахаров?</h1>
<p class='sub'>Классификация измеренных спектров. Разделение по физическим лункам, проверка при меньшем времени накопления сигнала.</p></div></header>
<main><div class='grid'>
<div class='card'><span>Лунок в обучении / тесте</span><strong>{data['train_wells']} / {data['test_wells']}</strong><span>Повторы одной лунки не пересекают границу</span></div>
<div class='card'><span>Balanced accuracy · обычные измерения</span><strong>{logistic_high:.1%}</strong><span>Логистическая регрессия</span></div>
<div class='card'><span>Balanced accuracy · слабый сигнал</span><strong>{logistic_low:.1%}</strong><span>Те же отложенные лунки</span></div></div>
<section><h2>Сравнение моделей</h2><div class='scroll'><table><thead><tr><th>Модель</th><th>Данные проверки</th><th>Balanced accuracy</th><th>F1</th><th>ROC AUC</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class='note'>«Частый класс» всегда предсказывает наличие глюкозы. Из-за дисбаланса классов его F1 выглядит высоким; balanced accuracy показывает, что модель не различает классы. Все модели обучены только на измерениях с большим временем накопления.</p></section>
<section><h2>Что сравнивается</h2><p>Бинарная метка: в рецептуре лунки указано больше 0 мкл раствора глюкозы. Спектры обрезаны до 400–1800 см⁻¹ и нормализованы отдельно для каждого измерения (SNV). Масштабирование признаков логистической регрессии обучается только на тренировочной части.</p>
<p>Тестовые лунки не встречаются при обучении. Для них сравниваются измерения с экспозицией 5 с и 0,5 с. Это проверка переноса на более шумный сигнал тех же рецептур, а не доказательство работы на новом приборе или неизвестном веществе.</p></section>
<div class='split'><section><h2>Средние спектры</h2><img src='mean_spectra.png' alt='Средние спектры обучающей выборки'></section>
<section><h2>Ошибки выбранной модели</h2><img src='confusion_logistic_low.png' alt='Матрица ошибок логистической регрессии на слабом сигнале'></section></div>
<section><h2>Вывод</h2><p>Линейная модель сохранила высокую разделимость на короткой экспозиции и в этом опыте оказалась устойчивее случайного леса. Это повод проверять простые базовые методы до усложнения архитектуры. Следующий эксперимент — разделение по дню приготовления или прибору, если появятся независимые партии измерений.</p>
<p class='note'>Данные: <a href='https://github.com/Alvaro-FG/Raman_Sugars'>Raman_Sugars, Alvaro Fernandez Galiana et al.</a> Сырые спектры в этом репозитории не публикуются; скрипт загрузки закреплён на конкретной ревизии источника.</p></section>
</main></body></html>"""
    (OUT / "report.html").write_text(page, encoding="utf-8")
    print("Wrote", OUT / "report.html")


if __name__ == "__main__":
    main()
