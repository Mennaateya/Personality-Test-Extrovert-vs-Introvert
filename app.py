# streamlit_personality_test.py
"""
Enhanced Streamlit app with bilingual UI (Arabic / English), multiple pages, and Plotly visualizations.
Pages:
- Test (simple personality questionnaire + prediction)
- Models Performance (metrics table + interactive Plotly charts for each saved model)
- Insights (stylish, content-rich page about Introverts vs Extroverts with Plotly charts)

Keeps backward compatibility with your Files/ pickles and provides graceful fallbacks.
Run: streamlit run streamlit_personality_test.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import plotly.express as px

# ------------------ Configuration ------------------
st.set_page_config(page_title="Personality Quick-Test", layout="wide")
FILES_DIR = 'Files'
NUMERIC_COLS = ['Time_spent_Alone','Stage_fear','Social_event_attendance','Going_outside','Drained_after_socializing','Friends_circle_size','Post_frequency']
NORM_COLS = ['Going_outside','Social_event_attendance','Post_frequency','Friends_circle_size']
SKEWED_COLS = ['Stage_fear','Drained_after_socializing']
ROBUST_COL = ['Time_spent_Alone']

# ------------------ Utils ------------------
@st.cache_resource
def load_pickle(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

@st.cache_resource
def find_model_in_files(files_dir=FILES_DIR):
    candidates = [
        'BestModel.pkl', 'RandomForestClassifier.pkl', 'XGBoost.pkl', 'XGBoostClassifier.pkl',
        'LightGBM.pkl', 'LGBMClassifier.pkl', 'CatBoost.pkl', 'CatBoostClassifier.pkl',
        'ExtraTreesClassifier.pkl', 'DecisionTreeClassifier.pkl', 'LogisticRegression.pkl',
        'KNeighborsClassifier.pkl', 'GaussianNB.pkl', 'AdaBoost.pkl', 'Bagging.pkl'
    ]
    for c in candidates:
        p = Path(files_dir) / c
        if p.exists():
            return str(p)
    all_pkls = list(Path(files_dir).glob('*.pkl'))
    preprocessors = ['IterativeImputer.pkl','StandardScaler.pkl','yeo-johnson.pkl','RobustScaler.pkl',
                     'Label_encoding_Stage_fear.pkl','Label_encoding_Drained_after_socializing.pkl','LabelEncoder_y.pkl',
                     'X_test.pkl','y_test.pkl','results_dict.pkl']
    for p in all_pkls:
        if p.name not in preprocessors:
            return str(p)
    return None

@st.cache_data
def load_results_dict(path=os.path.join(FILES_DIR, 'results_dict.pkl')):
    return load_pickle(path)

# ------------------ Load artifacts ------------------
le_stage = load_pickle(os.path.join(FILES_DIR, 'Label_encoding_Stage_fear.pkl'))
le_drained = load_pickle(os.path.join(FILES_DIR, 'Label_encoding_Drained_after_socializing.pkl'))
le_y = load_pickle(os.path.join(FILES_DIR, 'LabelEncoder_y.pkl'))
imputer = load_pickle(os.path.join(FILES_DIR, 'IterativeImputer.pkl'))
standard = load_pickle(os.path.join(FILES_DIR, 'StandardScaler.pkl'))
yeo = load_pickle(os.path.join(FILES_DIR, 'yeo-johnson.pkl'))
robust = load_pickle(os.path.join(FILES_DIR, 'RobustScaler.pkl'))
model_path = find_model_in_files()
model = load_pickle(model_path) if model_path else None
results_dict = load_results_dict()

# ------------------ Translations ------------------
STR = {
    'en':{
        'title': 'Simple Personality Test — Introvert vs Extrovert',
        'subtitle': 'Quick questionnaire and prediction',
        'nav_test': 'Test',
        'nav_models': 'Models Performance',
        'nav_insights': 'Insights',
        'submit': 'Predict',
        'time_alone': 'Time spent alone (hours)',
        'stage_fear': 'Stage fear?',
        'social_att': 'Social event attendance (times per month)',
        'going_outside': 'Going outside frequency (times per week)',
        'drained_after': 'Feel drained after socializing?',
        'friends_size': 'Close friends (count)',
        'post_freq': 'Post frequency (posts per week)',
        'model_missing': 'Model files not found — showing demo heuristic result',
        'prediction': 'Prediction',
        'confidence': 'Confidence (approx)',
        'feature_imp': 'Top features (model importance)',
        'save_csv': 'Export results as CSV',
        'download': 'Download',
        'no_results': 'No saved results to show.',
        'insight_title': 'Introvert vs Extrovert — Quick Guide',
        'insight_text': 'A concise, friendly explanation about traits and tips.'
    },
    'ar':{
        'title': 'اختبار شخصي بسيط — انطوائي أم اجتماعي',
        'subtitle': 'استبيان سريع وتنبؤ',
        'nav_test': 'الاختبار',
        'nav_models': 'أداء النماذج',
        'nav_insights': 'معلومات ونصايح',
        'submit': 'تنبؤ',
        'time_alone': 'الوقت في العزل (ساعات)',
        'stage_fear': 'الخوف من المسرح؟',
        'social_att': 'حضور المناسبات الاجتماعية (مرات/شهر)',
        'going_outside': 'الخروج (مرات/أسبوع)',
        'drained_after': 'بتحس بتعب بعد الاختلاط؟',
        'friends_size': 'عدد الأصدقاء المقربين',
        'post_freq': 'معدل النشر (منشورات/أسبوع)',
        'model_missing': 'ملفات النماذج غير موجودة — عرض نتيجة تجريبية',
        'prediction': 'التنبؤ',
        'confidence': 'الثقة (تقريبية)',
        'feature_imp': 'أهم المميزات (حسب النموذج)',
        'save_csv': 'تصدير النتائج كـ CSV',
        'download': 'تحميل',
        'no_results': 'لا يوجد نتائج محفوظة للعرض.',
        'insight_title': 'انطوائي vs اجتماعي — دليل سريع',
        'insight_text': 'شرح مختصر وودود عن الصفات ونصايح.'
    }
}

# ------------------ Layout & Navigation ------------------
lang = st.sidebar.selectbox('Language / اللغة', options=['Arabic / العربية','English'], index=0)
lang_code = 'ar' if lang.startswith('Arabic') else 'en'
T = STR[lang_code]

st.title(T['title'])
st.caption(T['subtitle'])

page = st.sidebar.radio('', options=[T['nav_test'], T['nav_models'], T['nav_insights']])

# ------------------ Test Page ------------------
if page == T['nav_test']:
    with st.form(key='input_form'):
        col1, col2, col3 = st.columns(3)
        with col1:
            time_alone = st.number_input(T['time_alone'], min_value=0.0, max_value=24.0, value=4.0, step=0.5)
            stage_fear = st.selectbox(T['stage_fear'], ['No','Yes'] if lang_code=='en' else ['لا','نعم'])
            social_att = st.slider(T['social_att'], 0.0, 30.0, 3.0)
        with col2:
            going_outside = st.slider(T['going_outside'], 0.0, 21.0, 4.0)
            drained_after = st.selectbox(T['drained_after'], ['No','Yes'] if lang_code=='en' else ['لا','نعم'])
            friends_size = st.number_input(T['friends_size'], min_value=0, max_value=100, value=5, step=1)
        with col3:
            post_freq = st.slider(T['post_freq'], 0.0, 50.0, 1.0)
            st.markdown('---')
            submitted = st.form_submit_button(T['submit'])

    if submitted:
        # normalize Yes/No across languages to 'Yes'/'No' for encoders
        if lang_code == 'ar':
            stage_fear_in = 'Yes' if stage_fear in ['نعم','Yes'] else 'No'
            drained_in = 'Yes' if drained_after in ['نعم','Yes'] else 'No'
        else:
            stage_fear_in = stage_fear
            drained_in = drained_after

        input_df = pd.DataFrame([{
            'Time_spent_Alone': time_alone,
            'Stage_fear': stage_fear_in,
            'Social_event_attendance': social_att,
            'Going_outside': going_outside,
            'Drained_after_socializing': drained_in,
            'Friends_circle_size': friends_size,
            'Post_frequency': post_freq
        }])

        # encode
        if le_stage is not None and le_drained is not None:
            try:
                input_df['Stage_fear'] = le_stage.transform(input_df['Stage_fear'])
                input_df['Drained_after_socializing'] = le_drained.transform(input_df['Drained_after_socializing'])
            except Exception:
                input_df['Stage_fear'] = input_df['Stage_fear'].map({'No':0,'Yes':1})
                input_df['Drained_after_socializing'] = input_df['Drained_after_socializing'].map({'No':0,'Yes':1})
        else:
            input_df['Stage_fear'] = input_df['Stage_fear'].map({'No':0,'Yes':1})
            input_df['Drained_after_socializing'] = input_df['Drained_after_socializing'].map({'No':0,'Yes':1})

        # impute & transform
        try:
            if imputer is not None:
                input_df[NUMERIC_COLS] = imputer.transform(input_df[NUMERIC_COLS])
        except Exception:
            pass
        try:
            if standard is not None:
                input_df[NORM_COLS] = standard.transform(input_df[NORM_COLS])
        except Exception:
            pass
        try:
            if yeo is not None:
                input_df[SKEWED_COLS] = yeo.transform(input_df[SKEWED_COLS])
        except Exception:
            pass
        try:
            if robust is not None:
                input_df[ROBUST_COL] = robust.transform(input_df[ROBUST_COL])
        except Exception:
            pass

        # predict
        if model is not None:
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_df[NUMERIC_COLS] if hasattr(model, 'n_features_in_') else input_df.values.reshape(1,-1))
                    p_intro = proba[0][1] if proba.shape[1]==2 else proba[0].max()
                else:
                    p_intro = None
                pred = model.predict(input_df[NUMERIC_COLS] if hasattr(model, 'n_features_in_') else input_df.values.reshape(1,-1))
                if le_y is not None:
                    try:
                        pred_label = le_y.inverse_transform(pred)[0]
                    except Exception:
                        pred_label = 'Introvert' if int(pred[0])==1 else 'Extrovert'
                else:
                    pred_label = 'Introvert' if int(pred[0])==1 else 'Extrovert'

                st.success(f"{T['prediction']}: **{pred_label}**")
                if p_intro is not None:
                    st.info(f"{T['confidence']}: {round(float(p_intro)*100,2)}%")

                if hasattr(model, 'feature_importances_'):
                    fi = pd.DataFrame({'feature': NUMERIC_COLS, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
                    st.subheader(T['feature_imp'])
                    st.table(fi.set_index('feature'))

            except Exception as e:
                st.error('Model failed to predict: ' + str(e))
        else:
            # fallback heuristic
            score = 0
            score += (time_alone / 24.0) * 2
            score += (1 if stage_fear_in=='Yes' else 0) * 1.0
            score += (0 if drained_in=='No' else 1.0)
            score += max(0, (5 - friends_size)/5.0)
            score += max(0, (3 - post_freq)/3.0)
            score = score / 6.0
            pred_label = 'Introvert' if score > 0.5 else 'Extrovert'
            st.warning(T['model_missing'])
            st.success(f"{T['prediction']} (demo): **{pred_label}** — score: {round(score,2)}")

# ------------------ Models Performance Page ------------------
elif page == T['nav_models']:
    st.header(T['nav_models'])
    if results_dict is None:
        st.info(T['no_results'])
        st.write('Place `results_dict.pkl` in the Files/ folder (created by training script).')
    else:
        # Build metrics dataframe
        rows = []
        for name, d in results_dict.items():
            m = d.get('metrics', {})
            rows.append({
                'Model': name,
                'Accuracy': m.get('Accuracy', np.nan),
                'Precision': m.get('Precision', np.nan),
                'Recall': m.get('Recall', np.nan),
                'F1': m.get('F1', np.nan)
            })
        metrics_df = pd.DataFrame(rows).sort_values('Accuracy', ascending=False)
        st.dataframe(metrics_df, use_container_width=True)

        # Plotly: bar chart of metrics
        metric_to_plot = st.selectbox('Metric', options=['Accuracy','Precision','Recall','F1'])
        fig = px.bar(metrics_df, x='Model', y=metric_to_plot, title=f'{metric_to_plot} per model', text=metric_to_plot)
        fig.update_layout(xaxis_tickangle=-45, yaxis_title=metric_to_plot)
        st.plotly_chart(fig, use_container_width=True)

        # Confusion matrix selector
        sel_model = st.selectbox('Show confusion matrix for', options=list(results_dict.keys()))
        cm = results_dict[sel_model]['confusion_matrix']
        class_names = results_dict[sel_model].get('class_names', ['0','1'])
        if cm is not None:
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            fig2 = px.imshow(cm_df, text_auto=True, title=f'Confusion Matrix — {sel_model}')
            st.plotly_chart(fig2, use_container_width=True)

# ------------------ Insights Page ------------------
elif page == T['nav_insights']:
    st.header(T['insight_title'])
    left, right = st.columns([2,1])
    with left:
        st.markdown(f"**{T['insight_text']}**")
        st.markdown('''
        - **Traits of Introverts:** spending more time alone, feel drained after socializing, prefer deeper 1:1 conversations, lower posting frequency.
        - **Traits of Extroverts:** larger social circles, energized by social events, higher posting/activity frequency.

        **Tips:**
        - Introverts: schedule quiet recovery time after social events; practice small social exposures gradually.
        - Extroverts: make space for reflection and active listening; ensure balanced online/offline interactions.
        ''')
    with right:
        # Image from Unsplash (stock photo) — using container width to avoid deprecation warning
        st.image('https://images.unsplash.com/photo-1755127761412-2934cc990b49?q=80&w=686&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D', caption='Balance is key', use_container_width=True)

    # Create illustrative Plotly charts (synthetic / demo if no dataset provided)
    st.subheader('Examples — Feature distributions')
    # Try to load saved X_test.pkl & y_test.pkl for real charts
    X_test_pickle = load_pickle(os.path.join(FILES_DIR, 'X_test.pkl'))
    y_test_pickle = load_pickle(os.path.join(FILES_DIR, 'y_test.pkl'))
    if X_test_pickle is not None and y_test_pickle is not None:
        Xdf = X_test_pickle.copy()
        if isinstance(y_test_pickle, (list, tuple, pd.Series, np.ndarray)):
            yser = pd.Series(y_test_pickle, name='Personality')
        else:
            yser = pd.Series(y_test_pickle, name='Personality')
        # if encoded numeric, try to decode
        if le_y is not None and yser.dtype != object:
            try:
                yser = pd.Series(le_y.inverse_transform(yser.astype(int)), name='Personality')
            except Exception:
                pass
        demo_df = pd.concat([Xdf.reset_index(drop=True), yser.reset_index(drop=True)], axis=1)
        # Plotly pie: Personality distribution
        dist = demo_df['Personality'].value_counts().reset_index()
        dist.columns = ['Personality','Count']
        fig_pie = px.pie(dist, names='Personality', values='Count', title='Personality Distribution')
        st.plotly_chart(fig_pie, use_container_width=True)

        # Boxplot: Time spent alone by Personality
        if 'Time_spent_Alone' in demo_df.columns:
            fig_box = px.box(demo_df, x='Personality', y='Time_spent_Alone', title='Time Spent Alone by Personality')
            st.plotly_chart(fig_box, use_container_width=True)

        # Scatter: Friends vs Social attendance colored by Personality
        if 'Friends_circle_size' in demo_df.columns and 'Social_event_attendance' in demo_df.columns and 'Post_frequency' in demo_df.columns:
            size_series = demo_df['Post_frequency'].astype(float).copy()
            # normalize to positive sizes for plotly
            if size_series.max() - size_series.min() == 0:
                sizes = np.full_like(size_series, fill_value=10.0, dtype=float)
            else:
                sizes = (size_series - size_series.min()) / (size_series.max() - size_series.min())
                sizes = sizes * 30 + 5  # scale to [5,35]
            fig_scat = px.scatter(demo_df, x='Friends_circle_size', y='Social_event_attendance', color='Personality',
                                  size=sizes, size_max=40, hover_data=demo_df.columns, title='Friends vs Social Attendance')
            st.plotly_chart(fig_scat, use_container_width=True)
    else:
        st.info('Dataset previews not found. To show real charts place X_test.pkl and y_test.pkl in Files/.')
        # create synthetic example chart
        synth = pd.DataFrame({
            'Personality': np.random.choice(['Introvert','Extrovert'], size=200, p=[0.45,0.55]),
            'Time_spent_Alone': np.random.gamma(2,2,size=200),
            'Friends_circle_size': np.random.poisson(5,size=200),
            'Social_event_attendance': np.random.poisson(3,size=200),
            'Post_frequency': np.random.poisson(2,size=200)
        })
        dist = synth['Personality'].value_counts().reset_index()
        dist.columns = ['Personality','Count']
        fig_pie = px.pie(dist, names='Personality', values='Count', title='Personality (synthetic)')
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_box = px.box(synth, x='Personality', y='Time_spent_Alone', title='Time Spent Alone (synthetic)')
        st.plotly_chart(fig_box, use_container_width=True)

        # make scatter with safe positive sizes
        size_series = synth['Post_frequency'].astype(float).copy()
        if size_series.max() - size_series.min() == 0:
            sizes = np.full_like(size_series, fill_value=10.0, dtype=float)
        else:
            sizes = (size_series - size_series.min()) / (size_series.max() - size_series.min())
            sizes = sizes * 30 + 5
        fig_scat = px.scatter(synth, x='Friends_circle_size', y='Social_event_attendance', color='Personality',
                              size=sizes, size_max=40, hover_data=['Post_frequency'], title='Friends vs Social Attendance (synthetic)')
        st.plotly_chart(fig_scat, use_container_width=True)


