CREATE DATABASE IF NOT EXISTS Instagram;

USE Instagram;

-- creating table with constraints
CREATE TABLE User(
	ID INT PRIMARY KEY, -- Primary key constraint makes a column not null and unqiue, generally only one column is called primary key 
    Name VARCHAR(30) NOT NULL, -- not null constraint 
    Email VARCHAR(50) UNIQUE, -- unique constraint 
    Age INT,
    Followers INT DEFAULT 0,
    Following INT DEFAULT 0,  -- default constraint 
    CONSTRAINT CHECK (Age >= 16) -- check constraint
);

-- A foreign key is a field (or column) in one table that uniquely links to the primary key of another table
-- PK CANNOT HAVE NULL OR DUPLICATE VALUES BUT FK CAN

-- creating one more table and then linking it using foriegn key concept 
CREATE TABLE Posts(
	ID INT PRIMARY KEY,
    Content VARCHAR(30) NOT NULL,
    User_ID INT,
    FOREIGN KEY (User_ID) REFERENCES User(ID) -- USER ID = foreign key --> Posts table and the primary key --> User table, used for linking 
);

-- Inserting values in Table 
INSERT INTO User
(ID,Name,Email,Age,Followers,Following)
Values
(1,"adam","adam@yahoo.in",16,123,145),
(2,"bob","bob@gmail.com",17,200,200),
(3,"casey","casey@gmail.com",18,300,306),
(4,"donald","donald@gmail.com",19,400,105);

-- Important Clauses to view data in SQL (VVVVVIMP):-

-- 1. SELECT (USED TO VIEW THINGS FROM A TABLE, VVVVVVIMP AS IT IS USED ALMOST EVERYWHERE)

SELECT * FROM User; -- Show everything
SELECT Name,ID,Email FROM User; -- Show certain things only 
SELECT DISTINCT Name, Age FROM User; -- using DISTINCT keyword shows only unique values in all/selected columns

-- 2. WHERE CLAUSE (TO DEFINE SOME CONDITIONS, USED WITH LOGICAL/ARITHMATIC/COMPARISION OPERATORS, USED WITH SELECT)

SELECT * FROM User WHERE Followers >= 200 AND Following >= 200; -- AND 
SELECT * FROM User WHERE Followers >= 200 OR Following >= 200; -- OR 
SELECT * FROM User WHERE Age BETWEEN 16 AND 18; -- BETWEEN 
SELECT * FROM User WHERE Email IN ("donald@gmail.com","bob@gmail.com","ABC@GMAIL.COM"); -- IN
SELECT * FROM User WHERE Email NOT IN ("donald@gmail.com","bob@gmail.com","ABC@GMAIL.COM"); -- NOT

-- 3. LIMIT CLAUSE (SETS AN UPPER LIMIT ON HOW MANY ROWS/TUPLES TO RETURN, USED WITH SELECT)

SELECT * FROM User LIMIT 2; -- shows only first 2 entries although we have more than 2 in our User table
SELECT * FROM User Where Age > 15 LIMIT 2; -- can use with other clauses like WHERE aswell

-- 4. ORDER BY CLAUSE (SORT A TABLE/ROWS IN ASC OR DESC ORDER, MANDATORY TO MENTION SORTING BASED ON WHICH COLUMN, USED WITH SELECT)

SELECT * FROM User ORDER BY Followers ASC; -- shows the table in asc order based on followers
SELECT * FROM User WHERE Age > 15 ORDER BY Followers DESC; -- shows the table in desc order based on followers, can be used with other clauses
SELECT * FROM User ORDER BY Followers; -- if no order is mentioned (asc or desc), then default is ascending order

-- 5. AGGREGATE FUNCTIONS (PERFORM A CALCULATION ON A SET OF VALUES AND RETURNS A SINGLE VALUE, USED ON ONE COLUMN, USED WITH SELECT)

SELECT max(Age) FROM User; -- using MAX()
SELECT avg(Age) FROM User; -- using AVG()
SELECT min(Followers) FROM User; -- using MIN()
SELECT sum(Following) FROM User; -- using SUM()
SELECT count(ID) FROM User WHERE Followers > 200; -- using COUNT(), ALSO THESE CAN BE USED WITH OTHER CLAUSES AS WELL

-- 6. GROUP BY CLAUSE (COLLECTS DATA FROM MULTIPLE ROWS AND GROUPS THE RESULT BY ONE OR MORE COLUMN, TRICKY, USED WITH SELECT AND AGG FUNCTION)
	  -- NOTE 1 : "GROUP BY" MAKES THE GROUP AND AGGREGATE FUNCTION DOES SOME WORK AND GIVES RESULT BY USING THE GROUPS
      -- NOTE 2 : WE CANT PRINT ANYTHING ELSE WITH THE AGG FUNCTION BUT ONLY THE COLUMN USED FOR MAKING THE GROUPS 

SELECT Age, COUNT(ID) FROM User GROUP BY Age; -- show age/id of users of each group. groups are made on the basis of age.
SELECT Age, MAX(Followers) FROM User GROUP BY Age; -- show age and max(followers) of each group. groups are made on the basis of age.

-- 7. HAVING CLAUSE (SIMILAR TO WHERE CLAUSE, APPLIES A CONDITION BUT AFTER GROUPING DONE BY GROUP BY CLAUSE, USED WITH SELECT AND GROUP BY)

SELECT Age, max(Followers) FROM User GROUP BY Age HAVING max(Followers) > 200; -- Using HAVING condition after grouping 

-- 8. LIKE CLAUSE (USED TO SEARCH FOR A SPECIFIED PATTERN IN A COLUMN, USES WILDCARDS '%' AND '_', USED WITH WHERE)
	-- NOTE 1 : '%' REPRESENTS ZERO, ONE, OR MULTIPLE CHARACTERS
	-- NOTE 2 : '_' REPRESENTS EXACTLY ONE SINGLE CHARACTER

SELECT * FROM User WHERE Name LIKE 'A%'; -- shows users whose name starts with 'A'
SELECT * FROM User WHERE Name LIKE '%a'; -- shows users whose name ends with 'a'
SELECT * FROM User WHERE Name LIKE '%john%'; -- shows users whose name contains 'john' anywhere in it
SELECT * FROM User WHERE Email LIKE '%@gmail.com'; -- shows users who have a gmail address
SELECT * FROM User WHERE Name LIKE '_a%'; -- shows users whose name has 'a' in the second position
SELECT * FROM User WHERE Name LIKE 'a_%_%'; -- shows users whose name starts with 'a' and is at least 3 characters long
SELECT * FROM User WHERE Name LIKE 'J__'; -- shows users whose name starts with 'J' and is exactly 3 characters long (like 'Jon' or 'Jay')

-- GENERAL ORDER FOR USING THESE CLAUSES IN ONE SINGLE QUERY
/*
	SELECT
    FROM
    WHERE
    GROUP BY
    HAVING
    ORDER BY
*/

-- TABLE QUERIES (IMP)

-- 1. UPDATE (TO UPDATE EXISTING ROWS, RUN SAFE COMMAND FIRST AND THEN UPDATE QUERY, GENERALLY USED WITH WHERE CLAUSE FOR CONDITION)

SET SQL_SAFE_UPDATES = 0; -- MANDATORY TO RUN THIS BEFORE UPDATING A TABLE
UPDATE User SET Followers = 500 WHERE Age > 18; -- USING UPDATE QUERY TO PUT FOLLOWERS = 500 FOR USERS WITH AGE 18+

-- 2. DELETE (TO DELETE EXISTING ROWS, GENERALLY USED WITH WHERE CLAUSE FOR CONDTION, WITHOUT A CONDIION IT WILL DELETE EVERYTHING)

DELETE FROM User WHERE Age = 16; -- USING WHERE CLAUSE WITH DELETE IS SAFE, WITHOUT IT ALL THE ROWS WILL BE DELETED

-- 3. ALTER (TO CHANGE SCHEMA, 5 PARTS - ADD COLUMN, DROP COLUMN, RENAME TABLE, CHANGE/RENAME COLUMN, MODIFY COLUMN)

ALTER TABLE User ADD COLUMN City VARCHAR(30) DEFAULT "Surat"; -- A. ADDING A COLUMN
ALTER TABLE User DROP COLUMN City; -- B. DROPPING COLUMN
ALTER TABLE User RENAME TO User_Data; -- C. RENAMING TABLE
ALTER TABLE User_Data CHANGE COLUMN Email Email_ID VARCHAR(100) UNIQUE; -- D. RENAMING COLUMN, ALWAYS NEED TO DEFINE NEW DTYPE, CONSTRAINT 
ALTER TABLE User_Data MODIFY Followers INT DEFAULT 5; -- E. MODIFYING COLUMN, ONLY CHANGES DATATYE OR CONSTRAINT 

-- 4. TRUNCATE (DELETE ALL THE DATA IN TABLE, DROP TABLE DELETES THE WHOLE TABLE)

TRUNCATE TABLE User_Data; -- SHOWS ERROR AS USERDATA TABLE IS LINKED TO THE POST TABLE WE CREATED USING FK
DROP TABLE Posts; -- FIRST WE DROP THIS TABLE I.E. DELETE IT COMPLETELY
TRUNCATE TABLE User_Data; -- NOW, THIS WILL SHOW NO ERROR

SELECT * FROM User_Data;

























