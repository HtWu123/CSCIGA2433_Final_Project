CREATE TABLE Company
(
    company_id integer not null auto_increment,
    company_name varchar(100),
    company_phone varchar(20),
    PRIMARY KEY (company_id)
);

CREATE TABLE CompanyAcct
(
    company_id integer not null,
    acct_id integer not null,
    PRIMARY KEY (company_id, acct_id),
    FOREIGN KEY (company_id) REFERENCES Company (company_id) ON DELETE CASCADE,
    FOREIGN KEY (acct_id) REFERENCES Account (acct_id) ON DELETE CASCADE
);

CREATE TABLE Billing
(
    billing_id integer not null auto_increment,
    billing_date date,
    billing_name varchar(100),
    billing_type varchar(50),
    billing_state varchar(50),
    billing_city varchar(50),
    billing_zip varchar(20),
    billing_address varchar(255),
    PRIMARY KEY (billing_id)
);


CREATE TABLE BillingAcct
(
    billing_id integer not null,
    acct_id integer not null,
    PRIMARY KEY (billing_id, acct_id),
    FOREIGN KEY (billing_id) REFERENCES Billing (billing_id) ON DELETE CASCADE,
    FOREIGN KEY (acct_id) REFERENCES Account (acct_id) ON DELETE CASCADE
);


CREATE TABLE Bill
(
    bill_id integer not null auto_increment,
    contract_id varchar(50),
    plan_id integer,
    date date,
    deductible decimal(10, 2),
    copay decimal(10, 2),
    PRIMARY KEY (bill_id),
    FOREIGN KEY (contract_id) REFERENCES Contract (contract_id),
    FOREIGN KEY (plan_id) REFERENCES Plan (plan_id)
);

CREATE TABLE Account
(
    acct_id integer not null auto_increment,
    start_date date,
    acct_state varchar(50),
    acct_city varchar(50),
    acct_zip varchar(20),
    acct_address varchar(255),
    PRIMARY KEY (acct_id)
);

CREATE TABLE CustomerAcct
(
    insured_ssn varchar(20) not null,
    acct_id integer not null,
    PRIMARY KEY (insured_ssn, acct_id),
    FOREIGN KEY (insured_ssn) REFERENCES Customer (customer_ssn) ON DELETE CASCADE,
    FOREIGN KEY (acct_id) REFERENCES Account (acct_id) ON DELETE CASCADE
);

CREATE TABLE Customer
(
    customer_ssn varchar(20) not null,
    f_name varchar(50),
    m_init varchar(1),
    l_name varchar(50),
    gender varchar(10),
    dob date,
    address varchar(255),
    city varchar(50),
    state varchar(50),
    phone varchar(20),
    PRIMARY KEY (customer_ssn)
);

CREATE TABLE Associate
(
    asso_ssn varchar(20) not null,
    asso_Fname varchar(50),
    asso_minit varchar(1),
    asso_Lname varchar(50),
    asso_phone varchar(20),
    asso_address varchar(255),
    PRIMARY KEY (asso_ssn)
);

CREATE TABLE AssociateContract
(
    asso_ssn varchar(20) not null,
    contract_id varchar(50) not null,
    PRIMARY KEY (asso_ssn, contract_id),
    FOREIGN KEY (asso_ssn) REFERENCES Associate (asso_ssn) ON DELETE CASCADE,
    FOREIGN KEY (contract_id) REFERENCES Contract (contract_id) ON DELETE CASCADE
);

CREATE TABLE Contract
(
    contract_id varchar(50) not null,
    policyholder_id varchar(20),
    plan_id integer,
    policy_term_year integer,
    PRIMARY KEY (contract_id),
    FOREIGN KEY (plan_id) REFERENCES Plan (plan_id),
    FOREIGN KEY (policyholder_id) REFERENCES Customer (customer_ssn) ON DELETE SET NULL
);

CREATE TABLE ContractInsured
(
    contract_id varchar(50) not null,
    insured_ssn varchar(20) not null,
    PRIMARY KEY (contract_id, insured_ssn),
    FOREIGN KEY (contract_id) REFERENCES Contract (contract_id) ON DELETE CASCADE,
    FOREIGN KEY (insured_ssn) REFERENCES Customer (customer_ssn) ON DELETE CASCADE
);

CREATE TABLE License
(
    license_num varchar(50) not null,
    asso_ssn varchar(20) not null,
    license_type varchar(50),
    issue_date date,
    expire_date date,
    PRIMARY KEY (license_num),
    FOREIGN KEY (asso_ssn) REFERENCES Associate (asso_ssn) ON DELETE CASCADE
);

CREATE TABLE Insured_info
(
    insured_ssn varchar(20) not null,
    region varchar(50),
    Urban_Rural varchar(50),
    income decimal(15, 2),
    education varchar(100),
    bmi decimal(5, 2),
    smoker boolean,
    alcohol_freq varchar(50),
    exercise_freq varchar(50),
    sleep_hours decimal(4, 2),
    hypertension boolean,
    diabetes boolean,
    copd boolean,
    cardiovascular boolean,
    cancer_history boolean,
    kidney_disease boolean,
    liver_disease boolean,
    arthritis boolean,
    mental_health boolean,
    chronic_count integer,
    systolic_bp integer,
    diastolic_bp integer,
    ldl decimal(10, 2),
    hba1c decimal(5, 2),
    visits_last_year integer,
    hospitalizations_last_year integer,
    3yrs integer,
    days_hospitalized_last integer,
    3yrs_medication integer,
    _count integer,
    proc_surgery integer,
    proc_consult_count integer,
    PRIMARY KEY (insured_ssn),
    FOREIGN KEY (insured_ssn) REFERENCES Customer (customer_ssn) ON DELETE CASCADE
);

CREATE TABLE Plan
(
    plan_id integer not null auto_increment,
    plan_name varchar(100),
    plan_description text,
    PRIMARY KEY (plan_id)
);

CREATE TABLE Risk_Evaluation
(
    insured_ssn varchar(20) not null,
    risk_score decimal(5, 2),
    PRIMARY KEY (insured_ssn),
    FOREIGN KEY (insured_ssn) REFERENCES Customer (customer_ssn) ON DELETE CASCADE
);

