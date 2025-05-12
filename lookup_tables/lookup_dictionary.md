# 📘 Lookup Tables Data Dictionary

This document describes the structure and intended use of the CSV files in the `/lookup_tables/` directory.

---

## 📊 `fmr_by_year.csv`

**Purpose:** Lookup Fair Market Rents (FMRs) by bedroom size, zip code, and year.

| Column      | Description                                      |
|-------------|--------------------------------------------------|
| `year`      | Calendar year for the FMR                        |
| `state`     | Two-letter state abbreviation                    |
| `county`    | County name                                      |
| `zip`       | ZIP code for location granularity (optional)    |
| `br_0`-`br_4`| Fair Market Rent for 0 to 4-bedroom units       |

---

## 💰 `income_limits.csv`

**Purpose:** Lookup HUD income limits by household size and geography.

| Column           | Description                                 |
|------------------|---------------------------------------------|
| `year`           | Year of the income limits                   |
| `state`          | State                                       |
| `county`         | County                                      |
| `household_size` | Size of the household (1–8 typically)       |
| `extremely_low`  | 30% of Area Median Income (AMI)             |
| `very_low`       | 50% of AMI                                  |
| `low_income`     | 80% of AMI                                  |

---

## 🏠 `subsidy_limit_matrix.csv`

**Purpose:** Reference maximum subsidy allowed by program and bedroom size.

| Column         | Description                                      |
|----------------|--------------------------------------------------|
| `program_type` | Program type (e.g., PBV, PBRA)                   |
| `bedroom_size` | Number of bedrooms in unit                       |
| `max_subsidy`  | Max allowable subsidy for that program/size      |
| `notes`        | Optional notes or source citation                |

---

## 📌 Notes

- All values are illustrative and should be replaced with current data from HUD, local PHAs, or state housing finance agencies.
- Intended for use in program logic, validation checks, and document generation.

