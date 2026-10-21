DROP VIEW IF EXISTS model_source;
DROP VIEW IF EXISTS house_features;

CREATE VIEW house_features AS
SELECT
    h.*,

    -- Age at the time of sale.
    YrSold - YearBuilt AS house_age_at_sale,
    YrSold - YearRemodAdd AS years_since_remodel,

    -- Combined basement and above-ground living area.
    TotalBsmtSF + GrLivArea AS total_living_area_sf,

    -- Half bathrooms count as 0.5.
    FullBath
        + 0.5 * HalfBath
        + BsmtFullBath
        + 0.5 * BsmtHalfBath AS total_bathrooms,

    -- Combined porch and deck area.
    WoodDeckSF
        + OpenPorchSF
        + EnclosedPorch
        + "3SsnPorch"
        + ScreenPorch AS total_outdoor_area_sf,

    -- Preserve NULL when the source value is unknown.
    CASE
        WHEN GarageArea > 0 THEN 1
        WHEN GarageArea = 0 THEN 0
        ELSE NULL
    END AS has_garage,

    CASE
        WHEN TotalBsmtSF > 0 THEN 1
        WHEN TotalBsmtSF = 0 THEN 0
        ELSE NULL
    END AS has_basement,

    CASE
        WHEN PoolArea > 0 THEN 1
        WHEN PoolArea = 0 THEN 0
        ELSE NULL
    END AS has_pool

FROM houses AS h;

-- Training rows with known sale prices.
CREATE VIEW model_source AS
SELECT
    f.*,
    l.SalePrice
FROM house_features AS f
JOIN house_price_labels AS l ON f.Id = l.Id;