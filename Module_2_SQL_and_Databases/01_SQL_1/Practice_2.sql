-- Practice Question 2

CREATE DATABASE IF NOT EXISTS College;

USE College;

CREATE TABLE Student(
	Roll_No INT PRIMARY KEY,
    Name VARCHAR(30),
    City VARCHAR(30),
    Marks INT NOT NULL
);

INSERT INTO Student
(Roll_No, Name, City, Marks)
VALUES
(110, "Adam", "Delhi", 76),
(08, "Bob", "Mumbai", 65),
(124, "Casey", "Surat", 94),
(112, "Duke", "Surat", 80);

SELECT * FROM Student;

-- A. SELECT ALL STUDENTS WHO SCORED 75+

SELECT * FROM Student WHERE Marks >= 75;

-- B. FIND NAMES OF ALL CITIES WHERE STUDENTS ARE FROM

SELECT City FROM Student GROUP BY City;

-- C. FIND MAX MARKS OF STUDENTS FROM EACH CITY

SELECT City, max(Marks) FROM Student GROUP BY City;

-- D. FIND AVERAGE MARKS OF THE CLASS

SELECT Avg(Marks) FROM Student;

-- E. ADD A NEW COLUMN GRADE SUCH THAT 80+ MARKS => GRADE O, 70-80 MARKS => GRADE A, 60-70 MARKS => GRADE B

ALTER TABLE Student ADD COLUMN Grade VARCHAR(2);
UPDATE Student SET Grade = "O" WHERE Marks >= 80;
UPDATE Student SET Grade = "A" WHERE Marks >= 70 AND Marks < 80;
UPDATE Student SET Grade = "B" WHERE Marks >= 60 AND Marks < 70;




