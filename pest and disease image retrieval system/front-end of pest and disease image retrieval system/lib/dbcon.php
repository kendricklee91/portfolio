<?

	if (realpath($_SERVER['SCRIPT_FILENAME']) == realpath(__FILE__))
		exit("No direct script access allowed");
		
		
	// ���� �Է�
	$localhost = "localhost";
	$dbid = "root";
	$password = "apmsetup";
	$database = "cbir"; //Database �̸�

	$conn = mysql_connect($localhost, $dbid, $password) or die($database."�� ������ �� �����ϴ�.");
	mysql_query("SET NAMES utf8");
	mysql_select_db($database);
?>