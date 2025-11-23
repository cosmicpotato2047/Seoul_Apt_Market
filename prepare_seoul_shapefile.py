import geopandas as gpd

IN_SHAPE = "data/SIG.shp"
OUT_SHAPE = "data/SEOUL_SIG.shp"

gdf = gpd.read_file(IN_SHAPE, encoding="cp949")

# 서울만 필터링: 시군구 코드가 11로 시작하는 행
seoul_gdf = gdf[gdf["SIG_CD"].str.startswith("11")]

# 저장 (shp는 총 4~6개 파일로 저장됨)
seoul_gdf.to_file(OUT_SHAPE, encoding="cp949")

print("서울 전용 shapefile 저장 완료:", OUT_SHAPE)
