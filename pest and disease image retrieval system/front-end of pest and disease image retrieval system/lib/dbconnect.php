<?
	if (realpath($_SERVER['SCRIPT_FILENAME']) == realpath(__FILE__))
		exit("No direct script access allowed");
		
	// ���� �Է�
	$localhost = "ir.sejong.ac.kr";
	$dbid = "root";
	$password = "bdlab203a";
	$database = "haklim_db"; //Database �̸�

	$conn = mysql_connect($localhost, $dbid, $password) or die($database."�� ������ �� �����ϴ�.");
	mysql_query("SET NAMES utf8");
	mysql_select_db($database);
?>