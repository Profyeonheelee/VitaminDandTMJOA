# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:44:26 2026

@author: USER
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import matplotlib.gridspec as gridspec

# 1. 데이터 불러오기
file_dir = r"C:\Users\USER\Desktop\2026 연구 VitaminD ESR CRP Prolo"
file_name = "TMJOA_VitaminD_read.csv"
file_path = os.path.join(file_dir, file_name)

df = pd.read_csv(file_path)

# 데이터 전처리 및 그룹 생성
df['Vitamin D Status'] = np.where(df['VITAMIND'] < 20, 'Deficiency\n(<20 ng/mL)', 'Sufficiency\n(≥20 ng/mL)')
df['VAS Group'] = np.where(df['VAS'] <= 5, 'VAS ≤5', 'VAS ≥6')

# 2. 전체 스타일 설정 (논문 규격)
sns.set_theme(style="ticks")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['axes.labelsize'] = 13

# 메인 Figure 및 GridSpec 설정 (2x2 구조 안에 A번의 Marginal Plot을 넣기 위해 세분화)
fig = plt.figure(figsize=(15, 14))
outer_gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.3)

# ----------------------------------------------------------------------
# [A] Scatter plot with Marginal Density Plots (상단 좌측)
# ----------------------------------------------------------------------
# A 영역을 다시 3x3 Grid로 쪼개어 메인 산점도와 주변부 밀도 그래프 배치
inner_gs_a = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outer_gs[0, 0], wspace=0.05, hspace=0.05)

ax_main = plt.subplot(inner_gs_a[1:, :-1])
ax_histx = plt.subplot(inner_gs_a[0, :-1], sharex=ax_main)
ax_histy = plt.subplot(inner_gs_a[1:, -1], sharey=ax_main)

# 메인 산점도 및 추세선
corr, p_val = spearmanr(df['VITAMIND'].dropna(), df['VAS'].dropna())
sns.regplot(data=df, x='VITAMIND', y='VAS', ax=ax_main, color='#1f77b4',
            scatter_kws={'alpha':0.5, 'edgecolor':'w', 's':35}, line_kws={'color':'#d62728', 'lw':2})
ax_main.set_xlabel("Serum Vitamin D Level (ng/mL)")
ax_main.set_ylabel("VAS Pain Intensity")

# 주변부 밀도 플롯 (Marginal Density)
sns.kdeplot(data=df, x='VITAMIND', ax=ax_histx, color='#1f77b4', fill=True, alpha=0.3, lw=1.5)
sns.kdeplot(data=df, y='VAS', ax=ax_histy, color='#1f77b4', fill=True, alpha=0.3, lw=1.5)

# 주변부 그래프 축 눈금 제거
ax_histx.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
ax_histy.tick_params(axis="both", which="both", bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
ax_histx.set_ylabel("")
ax_histy.set_xlabel("")
sns.despine(ax=ax_histx, left=True, bottom=True)
sns.despine(ax=ax_histy, left=True, bottom=True)
sns.despine(ax=ax_main)

# 통계치 기입 및 타이틀 'A'
ax_main.text(0.05, 0.05, f"Spearman $r$ = {corr:.3f}\n$p$ = {p_val:.3f}", 
             transform=ax_main.transAxes, fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='0.8'))
ax_histx.set_title("A", loc='left', fontsize=18, fontweight='bold', pad=10)


# ----------------------------------------------------------------------
# [B] Distribution of VAS according to Vitamin D deficiency (상단 우측)
# ----------------------------------------------------------------------
ax_b = plt.subplot(outer_gs[0, 1])

sns.boxplot(data=df, x='Vitamin D Status', y='VAS', ax=ax_b, palette=['#7fb3d5', '#f9e79f'], width=0.4, showfliers=False)
sns.stripplot(data=df, x='Vitamin D Status', y='VAS', ax=ax_b, color='#2c3e50', alpha=0.4, jitter=0.15, size=4.5)

ax_b.set_title("B", loc='left', fontsize=18, fontweight='bold', pad=10)
ax_b.set_xlabel("Vitamin D Status")
ax_b.set_ylabel("VAS Pain Intensity")
sns.despine(ax=ax_b)


# ----------------------------------------------------------------------
# [C] Distribution of Vitamin D according to pain intensity group (하단 좌측)
# ----------------------------------------------------------------------
ax_c = plt.subplot(outer_gs[1, 0])

sns.boxplot(data=df, x='VAS Group', y='VITAMIND', ax=ax_c, palette=['#aabbca', '#e6b0aa'], width=0.4, showfliers=False, order=['VAS ≤5', 'VAS ≥6'])
sns.stripplot(data=df, x='VAS Group', y='VITAMIND', ax=ax_c, color='#2c3e50', alpha=0.4, jitter=0.15, size=4.5, order=['VAS ≤5', 'VAS ≥6'])

ax_c.set_title("C", loc='left', fontsize=18, fontweight='bold', pad=10)
ax_c.set_xlabel("Pain Intensity Group")
ax_c.set_ylabel("Serum Vitamin D Level (ng/mL)")
sns.despine(ax=ax_c)


# ----------------------------------------------------------------------
# [D] Heat map showing Spearman correlations (하단 우측)
# ----------------------------------------------------------------------
ax_d = plt.subplot(outer_gs[1, 1])

# 스크리닝 변수 선택 (분석용 주요 수치 변수 5~6개 구성 예시)
features_for_heatmap = ['VAS', 'VITAMIND', 'AGE', 'ESR', 'ZINC', 'DEP']
corr_matrix, _ = spearmanr(df[features_for_heatmap].dropna())
corr_df = pd.DataFrame(corr_matrix, index=features_for_heatmap, columns=features_for_heatmap)

# 이미지 예시처럼 대칭형 사각형 전체를 채우는 깔끔한 히트맵 스타일
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0,
            ax=ax_d, cbar_kws={"shrink": 0.85, "label": "Spearman correlation coefficient ($r$)"}, square=True)

ax_d.set_title("D", loc='left', fontsize=18, fontweight='bold', pad=10)
ax_d.set_xticklabels(ax_d.get_xticklabels(), rotation=45, ha='right')
ax_d.set_yticklabels(ax_d.get_yticklabels(), rotation=0)


# --- 최종 출력 및 저장 ---
# 아래 주석을 해제하면 고해상도(300DPI) 이미지 파일로 즉시 저장됩니다.
# plt.savefig(os.path.join(file_dir, "Supplementary_Figure_S2_Final.png"), dpi=300, bbox_inches='tight')

plt.show()