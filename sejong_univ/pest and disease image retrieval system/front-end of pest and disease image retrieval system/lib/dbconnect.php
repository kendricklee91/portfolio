<?
	if (realpath($_SERVER['SCRIPT_FILENAME']) == realpath(__FILE__))
		exit("No direct script access allowed");
		
	// 정보 입력
	$localhost = "ir.sejong.ac.kr";
	$dbid = "root";
	$password = "bdlab203a";
	$database = "haklim_db"; //Database 이름

	$conn = mysql_connect($localhost, $dbid, $password) or die($database."에 연결할 수 없습니다.");
	mysql_query("SET NAMES utf8");
	mysql_select_db($database);
?>