import streamlit as st
import pandas as pd
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.stats import rankdata
import time
from collections import defaultdict

st.set_page_config(layout="wide")
st.title("Predict Recession Performance")

df = pd.read_csv('Performance.csv')
bm = df[df.apply(lambda row: row.str.contains('VIXTWN', na=False).any(), axis=1)]
df = df[~df.index.isin(bm.index)]

config_order = ['Sector', 'Window', 'Selection', 'Regression', 'Scale', 'Lag', 'Response', 'Variable', 'Target', 'Goal', 'Method', 'Feature']
score_order = ['In-AUC', 'In-Brier', 'In-LL', 'Out-AUC', 'Out-Brier', 'Out-LL']
stat_order = ['Box Plot', 'Avg', 'Med', 'Std', 'Max', 'Min']

with st.expander("See all configurations", icon=':material/info:', expanded=False):
    max_config_vars = max([len(df[var].unique()) for var in config_order])

    header_cols = st.columns([0.2] + [1 for _ in range(len(config_order)-1)] + [1.2])
    header_cols[0].markdown("**\-**")
    for j, config in enumerate(config_order):
        header_cols[j + 1].markdown(f"**{config}**")

    count_cols = st.columns([0.2] + [1 for _ in range(len(config_order)-1)] + [1.2])
    count_cols[0].markdown("**#**")
    for j, config in enumerate(config_order):
        unique_values = sorted(df[config].unique(), key=lambda x: str(x))
        count_cols[j + 1].markdown(f"**({len(unique_values)})**")

    for i in range(max_config_vars):
        row_cols = st.columns([0.2] + [1 for _ in range(len(config_order)-1)] + [1.2])
        row_cols[0].markdown(f"**{i + 1}**")
        for j, config in enumerate(config_order):
            unique_values = sorted(df[config].unique(), key=lambda x: str(x))
            if i < len(unique_values):
                values = [value[0].upper() + value[1:] if isinstance(value, str) else value for value in unique_values]
                value = str(values[i])
                row_cols[j + 1].markdown(f"{value}")
            else:
                row_cols[j + 1].markdown("")

tab1, tab2, tab3 = st.tabs([':material/bar_chart: Scoreboard', ':material/table_chart: Table Overview', ':material/star: Top 10 Score'])

################################################################################

with tab1:
    col1, col2 = st.columns([0.175, 0.825], vertical_alignment="bottom")
    with col1:
        st.subheader('Configurations')
    with col2:
        change_vars = st.multiselect(
            '↓  Choose multiple variables for comparison.',
            options=config_order,
            default=['Sector', 'Window'],
            key="change_vars_tab1"
        )
        change_vars = sorted(change_vars, key=lambda x: config_order.index(x))

    if change_vars:
        max_columns = 4
        sub_col1, sub_col2 = st.columns([0.175, 0.825])
        with sub_col2:
            num_vars = len(change_vars)
            for i in range(0, num_vars, max_columns):
                cols = st.columns(max_columns)
                for j, var in enumerate(change_vars[i:i + max_columns]):
                    values = [value[0].upper() + value[1:] if isinstance(value, str) else value for value in sorted(df[var].unique())]
                    values = [str(value) for value in values]
                    if j < len(cols):
                        with cols[j]:
                            st.markdown(f":material/database: **{var} ({len(values)}):** {' / '.join(values)}")

    change_combinations = list(product(*(df[var].unique() for var in change_vars)))

    col1, col2 = st.columns([0.175, 0.825], vertical_alignment="bottom")
    with col1:
        st.subheader('Scores:')
    with col2:
        score_vars = st.multiselect(
            '↓  Select multiple performance metrics.',
            options=score_order,
            default=['In-AUC', 'Out-AUC'],
            key="score_vars_tab1"
        )
        score_vars = sorted(score_vars, key=lambda x: score_order.index(x))

    col1, col2 = st.columns([0.175, 0.825], vertical_alignment="bottom")
    with col1:
        st.subheader('Statistics:')
    with col2:
        stat_vars = st.multiselect(
            '↓  Choose statistical metrics to display.',
            options=stat_order,
            default=['Box Plot', 'Avg'],
            key="stat_vars_tab1"
        )
    stat_vars = sorted(stat_vars, key=lambda x: stat_order.index(x))


    col1, col2 = st.columns([0.95, 0.05], vertical_alignment='bottom')
    with col1:
        st.markdown(
            """
            <hr style="border-top: 5px solid #bbb; margin-mid: 0px;">
            """,
            unsafe_allow_html=True
        )
    with col2: 
        rerun_tab1 = False
        if st.button('Run', type='secondary', key='Rerun_tab1'):
            rerun_tab1 = True
        else:
            rerun_tab1 = False

    if rerun_tab1 == True: 
        results = {}
        for combo in change_combinations:
            filtered_df = df.copy()
            for var, value in zip(change_vars, combo):
                filtered_df = filtered_df[filtered_df[var] == value]
            
            result = {}
            for score_var in score_vars:
                result[score_var] = {}
                if 'Avg' in stat_vars:
                    result[score_var]['Avg'] = filtered_df[score_var].mean()
                if 'Med' in stat_vars:
                    result[score_var]['Med'] = filtered_df[score_var].median()
                if 'Std' in stat_vars:
                    result[score_var]['Std'] = filtered_df[score_var].std()
                if 'Max' in stat_vars:
                    result[score_var]['Max'] = filtered_df[score_var].max()
                if 'Min' in stat_vars:
                    result[score_var]['Min'] = filtered_df[score_var].min()
            
            results[combo] = result

        ranks = {}
        for score_var in score_vars:
            for stat_var in stat_vars:
                if stat_var == 'Box Plot':
                    continue
                if score_var in ['In-AUC', 'Out-AUC']:
                    ascending = False
                elif score_var in ['In-Brier', 'In-LL', 'Out-Brier', 'Out-LL']:
                    ascending = True
                else:
                    ascending = False

                if stat_var in ['Std']:
                    ascending = True

                stat_values = [results[combo][score_var][stat_var] for combo in change_combinations]
                ranked_values = rankdata(stat_values, method='min' if ascending else 'dense')

                if not ascending:
                    ranked_values = len(ranked_values) - ranked_values + 1

                for combo, rank in zip(change_combinations, ranked_values):
                    if combo not in ranks:
                        ranks[combo] = {}
                    if score_var not in ranks[combo]:
                        ranks[combo][score_var] = {}
                    ranks[combo][score_var][stat_var] = rank

        def md_head(score_var):
            return f'''
            <div style="display: inline-block; text-align: left;">
                <div style="font-size: 20px;margin-bottom: 4px;">
                    <span style="font-weight: 800;">{score_var}</span>
                </div>
            </div><br>
            '''

        def md_body(stat_var, score_var, stat_value, rank, total):
            midpoint = total / 2
            color = "red" if rank <= midpoint else "green"
            
            return f'''
            <div style="display: inline-block; text-align: left;">
                <div style="font-size: 16px; font-weight: 600;">
                    <span style="font-weight: 800;">{stat_var}.</span>
                </div>
                <div style="font-size: 32px; font-weight: 200; color: dimgray;margin-top: -12px;">
                    {stat_value:.3f}
                </div>
                <div style="font-size: 14px; font-weight: 800; color: {color}; margin-top: -12px;margin-bottom: 10px;">
                    # {rank}
                </div>
            </div><br>
            '''


        if len(change_vars) > 0 and len(score_vars) > 0:
            total_combinations = len(change_combinations)
            complete = 0
            bar = st.progress(complete, text='Calculating...')
            for i in range(0, total_combinations, 6):
                columns = st.columns(7)
                with columns[0]:
                    for var in change_vars:
                        st.markdown(f"##### {var}")
                
                for j, combo in enumerate(change_combinations[i:i + 7]):
                    if j < len(columns) - 1:
                        complete += 1/total_combinations
                        bar.progress(complete, text='Calculating...')
                        with columns[j + 1]:
                            for value in combo:
                                if isinstance(value, str):
                                    value = value[0].upper() + value[1:]
                                st.markdown(f"##### {value}")
                            for score_var in score_vars:
                                sub_cols = st.columns(2)
                                with sub_cols[0]:
                                    metric_html = ""
                                    metric_html += md_head(score_var)
                                    for stat_var in stat_vars:
                                        if stat_var == 'Box Plot':
                                            continue
                                        stat_value = results[combo][score_var][stat_var]
                                        rank = ranks[combo][score_var][stat_var]
                                        metric_html += md_body(stat_var, score_var, stat_value, rank, total_combinations)
                                    st.markdown(metric_html, unsafe_allow_html=True)
                                
                                if 'Box Plot' in stat_vars:
                                    with sub_cols[1]:
                                        fig, ax = plt.subplots(figsize=(4,20)) 
                                        sns.boxplot(data=df[df[change_vars[0]] == combo[0]][score_var], ax=ax,
                                                    color='gray', width=0.4,
                                                    linewidth=8, linecolor='black',
                                                    flierprops={'marker':'o', 'markersize':10},
                                                    boxprops={'linewidth':4})
                                        if score_var in ['In-AUC', 'Out-AUC']:
                                            ax.set_ylim(-0.05, 1.05)
                                        elif score_var in ['In-Brier', 'Out-Brier']:
                                            ax.set_ylim(0, df[score_var].max())
                                        elif score_var in ['In-LL', 'Out-LL']:
                                            ax.set_ylim(-0.5,df[score_var].max())
                                        ax.set_ylabel('')
                                        ax.spines['top'].set_linewidth(5)
                                        ax.spines['right'].set_linewidth(5)
                                        ax.spines['left'].set_linewidth(5)
                                        ax.spines['bottom'].set_linewidth(5)
                                        ax.grid(True, axis='y', linestyle='--', linewidth=6, color='gray', alpha=0.6)
                                        font_prop = FontProperties(weight='bold', size=36)
                                        for label in ax.get_yticklabels():
                                            label.set_fontproperties(font_prop)
                                        plt.tight_layout()
                                        st.pyplot(fig)

                                st.write('')
                                    
                st.markdown(
                    """
                    <hr style="border-top: 5px solid #bbb; margin-mid: 0px;">
                    """,
                    unsafe_allow_html=True
                )
            time.sleep(0.1)
            bar.empty()
        else:
            st.warning("Please select at least one configuration/score.")
    

################################################################################

with tab2:
    col1, col2, col3, col4 = st.columns([0.175, 0.275, 0.275, 0.275])
    with col1:
        st.subheader('Rows:')
    with col2:
        sub_col1, sub_col2, sub_col3 = st.columns([0.2, 0.7, 0.1])
        sub_col1.metric('Layer', '1')
        with sub_col2:
            default_row_1_index = config_order.index('Sector') if 'Sector' in config_order else 0
            row_var_1 = st.selectbox(
                '↓  Select the Row Layer 1 variable.',
                options=config_order + ['(None)'],
                index=default_row_1_index
            )
            if row_var_1 != '(None)':
                values = [value[0].upper() + value[1:] if isinstance(value, str) else value for value in df[row_var_1].unique()]
                values = [str(value) for value in values]
                st.markdown(f":material/database: **{row_var_1} ({len(values)}):** {' / '.join(values)}")
    with col3:
        sub_col1, sub_col2, sub_col3 = st.columns([0.2, 0.7, 0.1])
        sub_col1.metric('Layer', '2')
        with sub_col2:
            if row_var_1 == '(None)':
                options_2 = ['(None)']
                row_var_2 = st.selectbox(
                    '↓  Select the Row Layer 2 variable.',
                    options=options_2,
                    index=0
                )
            else:
                options_2 = [opt for opt in config_order if opt != row_var_1] + ['(None)']
                default_row_2_index = options_2.index('Window') if 'Window' in options_2 else len(options_2) - 1
                row_var_2 = st.selectbox(
                    '↓  Select the Row Layer 2 variable.',
                    options=options_2,
                    index=default_row_2_index
                )
            if row_var_2 != '(None)':
                values = [value[0].upper() + value[1:] if isinstance(value, str) else value for value in df[row_var_2].unique()]
                values = [str(value) for value in values]
                st.markdown(f":material/database: **{row_var_2} ({len(values)}):** {' / '.join(values)}")
    with col4:
        sub_col1, sub_col2, sub_col3 = st.columns([0.2, 0.7, 0.1])
        sub_col1.metric('Layer', '3')
        with sub_col2:
            if row_var_2 == '(None)':
                options_3 = ['(None)']
                row_var_3 = st.selectbox(
                    '↓  Select the Row Layer 3 variable.',
                    options=options_3,
                    index=0
                )
            else:
                options_3 = [opt for opt in config_order if opt != row_var_1 and opt != row_var_2] + ['(None)']
                default_row_3_index = len(options_3) - 1
                row_var_3 = st.selectbox(
                    '↓  Select the Row Layer 3 variable.',
                    options=options_3,
                    index=default_row_3_index
                )
            if row_var_3 != '(None)':
                values = [value[0].upper() + value[1:] if isinstance(value, str) else value for value in df[row_var_3].unique()]
                values = [str(value) for value in values]
                st.markdown(f":material/database: **{row_var_3} ({len(values)}):** {' / '.join(values)}")

    st.markdown('')
    col1, col2, col3, col4 = st.columns([0.175, 0.275, 0.275, 0.275])
    with col1:
        st.subheader('Columns:')
    with col2:
        sub_col1, sub_col2, sub_col3 = st.columns([0.2, 0.7, 0.1])
        sub_col1.metric('Layer', '1')
        with sub_col2:
            available_col_1_options = [opt for opt in config_order if opt not in [row_var_1, row_var_2, row_var_3]] + ['(None)']
            default_col_1_index = available_col_1_options.index('Method') if 'Method' in available_col_1_options else 0
            col_var_1 = st.selectbox(
                '↓  Select the Column Layer 1 variable.',
                options=available_col_1_options,
                index=default_col_1_index
            )
            if col_var_1 != '(None)':
                values = [value[0].upper() + value[1:] if isinstance(value, str) else value for value in df[col_var_1].unique()]
                values = [str(value) for value in values]
                st.markdown(f":material/database: **{col_var_1} ({len(values)}):** {' / '.join(values)}")
    with col3:
        sub_col1, sub_col2, sub_col3 = st.columns([0.2, 0.7, 0.1])
        sub_col1.metric('Layer', '2')
        with sub_col2:
            if col_var_1 == '(None)':
                options_2 = ['(None)']
                col_var_2 = st.selectbox(
                    '↓  Select the Column Layer 2 variable.',
                    options=options_2,
                    index=0
                )
            else:
                available_col_2_options = [opt for opt in config_order if opt not in [row_var_1, row_var_2, row_var_3, col_var_1]] + ['(None)']
                default_col_2_index = available_col_2_options.index('Variable') if 'Variable' in available_col_2_options else len(available_col_2_options) - 1
                col_var_2 = st.selectbox(
                    '↓  Select the Column Layer 2 variable.',
                    options=available_col_2_options,
                    index=default_col_2_index
                )
            if col_var_2 != '(None)':
                values = [value[0].upper() + value[1:] if isinstance(value, str) else value for value in df[col_var_2].unique()]
                values = [str(value) for value in values]
                st.markdown(f":material/database: **{col_var_2} ({len(values)}):** {' / '.join(values)}")
    with col4:
        sub_col1, sub_col2, sub_col3 = st.columns([0.2, 0.7, 0.1])
        sub_col1.metric('Layer', '3')
        with sub_col2:
            if col_var_2 == '(None)':
                options_3 = ['(None)']
                col_var_3 = st.selectbox(
                    '↓  Select the Column Layer 3 variable.',
                    options=options_3,
                    index=0
                )
            else:
                available_col_3_options = [opt for opt in config_order if opt not in [row_var_1, row_var_2, row_var_3, col_var_1, col_var_2]] + ['(None)']
                default_col_3_index = len(available_col_3_options) - 1
                col_var_3 = st.selectbox(
                    '↓  Select the Column Layer 3 variable.',
                    options=available_col_3_options,
                    index=default_col_3_index
                )
            if col_var_3 != '(None)':
                values = [value[0].upper() + value[1:] if isinstance(value, str) else value for value in df[col_var_3].unique()]
                values = [str(value) for value in values]
                st.markdown(f":material/database: **{col_var_3} ({len(values)}):** {' / '.join(values)}")

    st.markdown('')

    col1, col2, col3 = st.columns([0.2, 0.7, 0.1], vertical_alignment="bottom")
    with col1:
        st.subheader('Seperate Variables:')
    with col3:
        if st.button('Clear'):
            st.session_state['sep_vars'] = ['(None)']
    with col2:
        def sep_update():
            if '(None)' in st.session_state.sep_vars and st.session_state.sep_vars[0] != '(None)':
                st.session_state['sep_vars'] = ['(None)']
            elif '(None)' in st.session_state.sep_vars and len(st.session_state.sep_vars) > 1:
                st.session_state.sep_vars.remove('(None)')
            elif len(st.session_state.sep_vars) == 0:
                st.session_state['sep_vars'] = ['(None)']
        available_sep_options = [opt for opt in config_order if opt not in [row_var_1, row_var_2, row_var_3, col_var_1, col_var_2, col_var_3]] + ['(None)']
        sep_vars = st.multiselect(
            '↓  Select variables that want to separately observe.',
            options=available_sep_options,
            default=st.session_state.get('sep_vars', ['(None)']),
            key='sep_vars',
            on_change=sep_update
        )

    st.markdown('')

    col1, col2 = st.columns([0.175, 0.825], vertical_alignment="bottom")
    with col1:
        st.subheader('Scores:')
    with col2:
        score_vars = st.multiselect(
            '↓  Select multiple performance metrics.',
            options=score_order,
            default=['In-AUC', 'Out-AUC'],
            key="score_vars_tab2"
        )
        score_vars = sorted(score_vars, key=lambda x: score_order.index(x))

    col1, col2 = st.columns([0.175, 0.825], vertical_alignment="bottom")
    with col1:
        st.subheader('Statistic:')
    with col2:
        stat_order_tab2 = ['Average', 'Median', 'Maximum', 'Minimum']
        stat_var = st.selectbox(
            '↓  Select statistical to display.',
            options=stat_order_tab2,
            index=0,
            key="stat_var_tab2"
        )

    col1, col2 = st.columns([0.95, 0.05], vertical_alignment='bottom')
    with col1:
        st.markdown(
            """
            <hr style="border-top: 5px solid #bbb; margin-mid: 0px;">
            """,
            unsafe_allow_html=True
        )
    with col2: 
        rerun_tab2 = False
        if st.button('Run', type='secondary', key='Rerun_tab2'):
            rerun_tab2 = True
        else:
            rerun_tab2 = False

    if rerun_tab2 == True: 
        complete = 0
        bar = st.progress(complete, text='Calculating...')

        if sep_vars != ['(None)']:
            sep_combinations = list(product(*(df[var].unique() for var in sep_vars)))
        else:
            sep_combinations = list(['(None)'])

        block_count = 0
        total_combinations = len(sep_combinations) * len(score_vars)
        for sep in sep_combinations:
            block_count += 1
            if sep != '(None)':
                values = [value[0].upper() + value[1:] if isinstance(value, str) else value for value in sep]
                values = [str(value) for value in values]
                st.markdown('### ' + f'{block_count}. Seperate Configuration: ' + ' $$\\times$$ '.join(map(str, values)))
            else:
                st.markdown('### 1. Seperate Configuration: (None)')
            subblock_count = 0
            for score_var in score_vars:
                complete += 1/total_combinations
                bar.progress(complete, text='Calculating...')
                subblock_count += 1
                main_col1, main_col2 = st.columns([0.01, 0.99])
                with main_col2:
                    st.markdown(
                        f"""
                        <h4 style="margin-bottom: -100px;">{block_count}-{subblock_count}. {score_var}</h4>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        """
                        <hr style="border-top: 2px solid #bbb; margin-bottom: 10px;">
                        """,
                        unsafe_allow_html=True
                    )
                    row_vars = [row_var_1, row_var_2, row_var_3]
                    row_vars = [var for var in row_vars if var != '(None)']

                    col_vars = [col_var_1, col_var_2, col_var_3]
                    col_vars = [var for var in col_vars if var != '(None)']

                    row_combinations = list(product(*(df[var].unique() for var in row_vars)))
                    col_combinations = list(product(*(df[var].unique() for var in col_vars)))

                    if row_var_1 != '(None)' and col_var_1 != '(None)':
                        col_num1, col_num2 = len(col_combinations), len(row_vars)
                        col_num = col_num1 + col_num2
                        
                        table_layer_1_length = defaultdict(int)
                        for combo in col_combinations:
                            table_layer_1_length[combo[0]] += 1
                        table_layer_1_length = [len(row_vars)] + list(table_layer_1_length.values())
                        table_layer_1 = st.columns(table_layer_1_length)
                        for i in range(len(table_layer_1_length)):
                            with table_layer_1[i]:
                                if i == 0:
                                    st.markdown(f"<div style='text-align: left;'><strong>{col_vars[0]}</strong></div>", unsafe_allow_html=True)
                                    st.markdown('<hr style="margin: 0px 0;">', unsafe_allow_html=True)
                                else:
                                    values = [value[0].upper() + value[1:] if isinstance(value, str) else value for value in df[col_vars[0]].unique()]
                                    values = [str(value) for value in values]
                                    st.markdown(f"<div style='text-align: center;'><strong>{values[i-1]}</strong></div>", unsafe_allow_html=True)
                                    st.markdown('<hr style="margin: 0px 0;">', unsafe_allow_html=True)

                        if len(col_vars) > 1:
                            table_layer_2_length = defaultdict(int)
                            for combo in col_combinations:
                                table_layer_2_length[(combo[0], combo[1])] += 1

                            table_layer_2_length = [len(row_vars)] + list(table_layer_2_length.values())
                            table_layer_2 = st.columns(table_layer_2_length)
                            table_layer_2_elements = [combo[1] for combo in col_combinations]

                            for i in range(len(table_layer_2_length)):
                                with table_layer_2[i]:
                                    if i == 0:
                                        st.markdown(f"<div style='text-align: left;'><strong>{col_vars[1]}</strong></div>", unsafe_allow_html=True)
                                        st.markdown('<hr style="margin: 0px 0;">', unsafe_allow_html=True)
                                    else:
                                        element = table_layer_2_elements[(i-1) * len(col_combinations) // (len(table_layer_2_length)-1)]
                                        element = element[0].upper() + element[1:] if isinstance(element, str) else element
                                        st.markdown(f"<div style='text-align: center;'><strong>{element}</strong></div>", unsafe_allow_html=True)
                                        st.markdown('<hr style="margin: 0px 0;">', unsafe_allow_html=True)

                        if len(col_vars) > 2:
                            table_layer_3_length = defaultdict(int)
                            for combo in col_combinations:
                                table_layer_3_length[(combo[0], combo[1], combo[2])] += 1

                            table_layer_3_length = [len(row_vars)] + list(table_layer_3_length.values())
                            table_layer_3 = st.columns(table_layer_3_length)
                            table_layer_3_elements = [combo[2] for combo in col_combinations]

                            for i in range(len(table_layer_3_length)):
                                with table_layer_3[i]:
                                    if i == 0:
                                        st.markdown(f'**{col_vars[2]}**')
                                    else:
                                        element = table_layer_3_elements[(i-1) * len(col_combinations) // (len(table_layer_3_length)-1)]
                                        element = element[0].upper() + element[1:] if isinstance(element, str) else element
                                        st.markdown(f"<div style='text-align: center;'><strong>{element}</strong></div>", unsafe_allow_html=True)
                                        st.markdown('<hr style="margin: 0px 0;">', unsafe_allow_html=True)
                        
                        table_length = defaultdict(int)
                        for combo in col_combinations:
                            table_length[tuple(combo[:len(col_vars)])] += 1
                        table_length = len(row_vars)*[1] + list(table_length.values())
                        table = st.columns(table_length)
                        for i in range(len(row_vars)):
                            with table[i]:
                                st.markdown(f"<div style='text-align: left;'><strong>{row_vars[i]}</strong></div>", unsafe_allow_html=True)
                                st.markdown('<hr style="margin: 0px 0;">', unsafe_allow_html=True)

                        table_length = defaultdict(int)
                        for combo in col_combinations:
                            table_length[tuple(combo[:len(col_vars)])] += 1
                        table_length = len(row_vars)*[1] + list(table_length.values())
                        table = st.columns(table_length)
                        for row_combo in range(len(row_combinations)):
                            for i in range(len(table_length)):
                                with table[i]:
                                    if i < len(row_vars):
                                        element = row_combinations[row_combo][i]
                                        element = element[0].upper() + element[1:] if isinstance(element, str) else element
                                        st.markdown(f"<div style='text-align: left; line-height: 2;'><strong>{element}</strong></div>", unsafe_allow_html=True)
                                    else:
                                        element_configuration = [str(sep)] if isinstance(sep, str) else [str(item) for item in sep]
                                        element_configuration += [str(item) for item in row_combinations[row_combo]] + [str(item) for item in col_combinations[i-len(row_vars)]]
                                        element_configuration = [item for item in element_configuration if item != '(None)']
                                        element_variable = [str(item) for item in sep_vars] + [str(item) for item in row_vars] + [str(item) for item in col_vars]
                                        element_variable = [item for item in element_variable if item != '(None)']
                                        condition = True
                                        filtered_df = df.copy()
                                        filtered_df.iloc[:, :12] = filtered_df.iloc[:, :12].astype(str)
                                        for var, config in zip(element_variable, element_configuration):
                                            condition = condition & (filtered_df[var] == config)
                                        filtered_df = filtered_df[condition]
                                        filtered_scores = filtered_df[score_var]
                                        if stat_var == 'Average':
                                            element_score = filtered_scores.mean()
                                        elif stat_var == 'Median':
                                            element_score = filtered_scores.median()
                                        elif stat_var == 'Maximum':
                                            element_score = filtered_scores.max()
                                        elif stat_var == 'Minimum':
                                            element_score = filtered_scores.min()
                                        st.markdown(f"<div style='text-align: center; line-height: 2;'>{element_score:.3f}</div>", unsafe_allow_html=True)

                        change_vars = row_vars + col_vars
                        filtered_vars = [item for item in config_order if item not in change_vars]
                        st.markdown(
                            """
                            <hr style="border-top: 2px solid #bbb; margin-top: 5px;">
                            """,
                            unsafe_allow_html=True
                        )
            st.markdown(
                """
                <hr style="border-top: 5px solid #bbb; margin-top: -25px;">
                """,
                unsafe_allow_html=True
            )
            time.sleep(0.5)
            bar.empty()

################################################################################

with tab3:
    st.subheader('Top 10 Scores')

    # Create 12 columns for filters
    st.markdown("#### Filters:")
    filter_columns = st.columns(12)
    filters = {}
    for i, var in enumerate(config_order):
        with filter_columns[i]:
            unique_values = sorted(df[var].unique())
            # Capitalize string values for display
            display_values = [value.capitalize() if isinstance(value, str) else value for value in unique_values]
            selected_values = st.multiselect(
                label=var,
                options=unique_values,
                default=unique_values,
                key=f'filter_{var}'
            )
            filters[var] = selected_values

    st.markdown("---")

    # Let the user select the score metric
    score_var = st.selectbox(
        'Select performance metric:',
        options=score_order,
        index=score_order.index('Out-AUC') if 'Out-AUC' in score_order else 0,
        key='score_var_tab3'
    )

    col1, col2 = st.columns([0.95, 0.05], vertical_alignment='bottom')
    with col1:
        st.markdown(
            """
            <hr style="border-top: 5px solid #bbb; margin-mid: 0px;">
            """,
            unsafe_allow_html=True
        )
    with col2:
        rerun_tab3 = False
        if st.button('Run', type='secondary', key='Rerun_tab3'):
            rerun_tab3 = True
        else:
            rerun_tab3 = False

    if rerun_tab3 == True:
        # Apply filters
        filtered_df = df.copy()
        for var in config_order:
            selected_values = filters[var]
            if selected_values:
                filtered_df = filtered_df[filtered_df[var].isin(selected_values)]
            else:
                # If no values are selected, include all values for this variable
                pass

        if filtered_df.empty:
            st.warning("No configurations match the selected filters.")
        else:
            # Decide whether to maximize or minimize based on the performance metric
            if score_var in ['In-AUC', 'Out-AUC']:
                ascending = False  # Maximize AUC
            elif score_var in ['In-Brier', 'In-LL', 'Out-Brier', 'Out-LL']:
                ascending = True  # Minimize Brier and LL
            else:
                ascending = False

            # Group the data by configurations and compute mean score
            grouped = filtered_df.groupby(config_order)[score_var].mean().reset_index()

            if grouped.empty:
                st.warning("No configurations match the selected filters.")
            else:
                # Sort configurations according to the mean score
                score_stat_sorted = grouped.sort_values(by=score_var, ascending=ascending)

                # Get top 10
                top_10 = score_stat_sorted.head(10)

                if len(top_10) < 10:
                    st.warning(f"Only {len(top_10)} configurations match the selected filters.")

                # Reset index and adjust for display
                top_10.reset_index(drop=True, inplace=True)
                top_10.index = top_10.index + 1  # Start index from 1
                top_10[score_var] = top_10[score_var].round(4)

                # Display the top 10 configurations
                st.write('### Top 10 Configurations')
                st.dataframe(top_10, use_container_width=True)