-- Assignment Question 

CREATE DATABASE IF NOT EXISTS Company;

USE Company;

CREATE TABLE Employee(
	Emp_ID INT PRIMARY KEY,
    First_Name VARCHAR(20),
    Last_Name VARCHAR(30),
    Department VARCHAR(20),
    Salary INT DEFAULT 0,
    HireDate DATE
);

INSERT INTO Employee
(Emp_ID, First_Name, Last_Name, Department, Salary, HireDate)
VALUES
(101, 'Alice', 'Johnson', 'IT', 6500, '2020-03-15'),
(102, 'Mark', 'Rivera', 'HR', 4800, '2019-07-22'),
(103, 'Sophia', 'Lee', 'Finance', 7200, '2021-01-10'),
(104, 'Daniel', 'Kim', 'IT', 5800, '2018-11-05'),
(105, 'Emma', 'Brown', 'Marketing', 5300, '2022-04-18'),
(106, 'Liam', 'Patel', 'Finance', 6900, '2020-09-29'),
(107, 'Olivia', 'Garcia', 'HR', 4600, '2017-06-30'),
(108, 'Noah', 'Thompson', 'IT', 7500, '2023-02-12'),
(109, 'Ava', 'Martinez', 'Marketing', 5100, '2019-12-02'),
(110, 'Ethan', 'Davis', 'Finance', 8000, '2016-05-14');

-- Q1. Write a query to display every employee and all their data

SELECT * FROM Employee;

-- Q2. List only the FirstName, LastName, and Salary of every employee.

SELECT First_Name, Last_Name, Salary FROM Employee;

-- Q3. Show all employees who work in the 'IT' department

SELECT * FROM Employee WHERE Department = "IT";

-- Q4. Retrieve employees with a salary greater than 6000

SELECT * FROM Employee WHERE Salary > 6000;

-- Q5. List all employees ordered by HireDate from newest to oldest.

SELECT * FROM Employee ORDER BY HireDate DESC;

-- Q6. Show a list of all unique departments present in the table

SELECT DISTINCT Department FROM Employee;

-- Q7. Find employees whose first name starts with ‘Aʼ.

-- Q7. Find employees whose first name starts with 'A'.

SELECT * FROM Employee WHERE First_Name LIKE 'A%';

-- Q8. Show employees whose salaries are between 4000 and 7000

SELECT * FROM Employee WHERE Salary BETWEEN 4000 AND 7000;

-- Q9. Find the average salary of all employees.

SELECT avg(Salary) FROM Employee;

-- Q10. List each department along with the number of employees, but only include departmentswith more than 3 employees.

SELECT Department, COUNT(Emp_ID) FROM Employee GROUP BY Department HAVING COUNT(Emp_ID) >= 3;

