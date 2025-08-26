import pandas as pd
import json
import os
from bs4 import BeautifulSoup
from collections import defaultdict
import re

def classify_canvas(canvas, parent_class_map, index=0):
    # get parent <article> and its classes
    article = canvas.find_parent('article')
    dtype = "unknown"
    parent_class = article.get('class', []) if article else []
    # 1. Try parent_class_map
    for key, val in parent_class_map.items():
        if key in parent_class:
            dtype = val
    # 2. If unknown, try heading text
    if dtype == "unknown" and article:
        heading = article.find("h3")
        if heading and "unweighted" in heading.text.lower():
            dtype = "unweighted"
        elif heading and "weighted" in heading.text.lower():
            dtype = "weighted"
    # 3. Fallback: index (first canvas = unweighted, second = weighted)
    if dtype == "unknown":
        dtype = "unweighted" if index == 0 else "weighted"
    return dtype

def extract_canvas_by_parent_class(soup, cid, parent_class_map, month):
    result_tables = []
    canvases = list(soup.find_all('canvas', id=cid))
    for idx, canvas in enumerate(canvases):
        dtype = classify_canvas(canvas, parent_class_map, idx)
        if canvas.has_attr('data-json') and canvas.has_attr('data-labels'):
            data_json = json.loads(canvas['data-json'])
            data_labels = json.loads(canvas['data-labels'])
            df = pd.DataFrame(data_json, index=data_labels)
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Brand'}, inplace=True)
            df['Month'] = month
            df['DataType'] = dtype
            result_tables.append(df)
    return result_tables

def extract_modal_tables(soup, month):
    modal_tables = []
    modals = soup.find_all('div', class_='m22__brand-modal--inner')
    for modal in modals:
        brand_title = modal.find('h4', class_='m22__brand-modal--title').text.strip()
        brand = brand_title.replace(' Availability by Endpoint Category:', '')
        mth = modal.find('span', class_='m22__brand-modal--month').text.strip()
        # Standardize month
        mth_fmt = month
        match = re.search(r'([A-Za-z]+)\s+(\d{4})', mth)
        if match:
            try:
                mth_fmt = pd.to_datetime(match.group(0)).strftime('%Y-%m')
            except Exception:
                pass
        categories = modal.find_all('div', class_='m22__brand-modal--category')
        for cat in categories:
            cat_name = cat.find('h5', class_='title').text.strip()
            cat_value = cat.find('p', class_='data').text.strip()
            modal_tables.append({
                'Brand': brand,
                'Month': mth_fmt,
                'Endpoint Category': cat_name,
                'Availability': cat_value,
                'FileMonth': month
            })
    return pd.DataFrame(modal_tables)

parent_class_maps = {
    'coreHoursBrand': {
        'core-hours-by-brand-unweighted': 'unweighted',
        'core-hours-by-brand-weighted': 'weighted',
    },
    'nonCoreHoursBrand': {
        'non-core-hours-by-brand-unweighted': 'unweighted',
        'non-core-hours-by-brand-weighted': 'weighted',
    },
    'aisAvailabilityBrand': {
        'ais-availability-by-brand-unweighted': 'unweighted',
        'ais-availability-by-brand-weighted': 'weighted',
    },
    'pisAvailabilityBrand': {
        'pis-availability-by-brand-unweighted': 'unweighted',
        'pis-availability-by-brand-weighted': 'weighted',
    }
}

canvas_ids = [
    'averageAvaliability',
    'weightedAvaliability',
    'coreHoursBrand',
    'nonCoreHoursBrand',
    'aisAvailabilityBrand',
    'pisAvailabilityBrand',
    'averageResponseTimeBrand',
    'successfulApiCallsBrand',
    'aisCallsBrand',
    'pisCallsBrand',
    'paymentRequestsBrand',
    'failedApiCallsBrand',
    'failedBusinessCallsBrand',
    'failedTechCallsBrand',
    'rejectedApiCallsBrand'
]

months = pd.date_range("2021-06-01", "2025-05-01", freq="MS").strftime("%Y-%m").tolist()
all_tables = defaultdict(list)
all_modals = []

for month in months:
    filename = f"{month}_standard.html"
    if not os.path.exists(filename):
        print(f"{filename} not found, skipping.")
        continue
    with open(filename, encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
        for cid in canvas_ids:
            if cid in parent_class_maps:
                dfs = extract_canvas_by_parent_class(soup, cid, parent_class_maps[cid], month)
                for df in dfs:
                    all_tables[cid].append(df)
            else:
                # Only one canvas expected for these
                canvas = soup.find('canvas', id=cid)
                if canvas and canvas.has_attr('data-json') and canvas.has_attr('data-labels'):
                    data_json = json.loads(canvas['data-json'])
                    data_labels = json.loads(canvas['data-labels'])
                    df = pd.DataFrame(data_json, index=data_labels)
                    df.reset_index(inplace=True)
                    df.rename(columns={'index': 'Brand'}, inplace=True)
                    df['Month'] = month
                    df['DataType'] = 'standard'
                    all_tables[cid].append(df)
        modal_df = extract_modal_tables(soup, month)
        if not modal_df.empty:
            all_modals.append(modal_df)

with pd.ExcelWriter('openbanking_all_months.xlsx') as writer:
    for cid, df_list in all_tables.items():
        big_df = pd.concat(df_list, ignore_index=True)
        sheet = cid[:28]
        big_df.to_excel(writer, sheet_name=sheet, index=False)
    if all_modals:
        big_modal = pd.concat(all_modals, ignore_index=True)
        big_modal.to_excel(writer, sheet_name='BrandEndpointAvail', index=False)

print("Exported all months to openbanking_all_months.xlsx")
