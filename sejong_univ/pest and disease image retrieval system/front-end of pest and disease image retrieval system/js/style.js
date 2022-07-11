var viewCnt=0;

/////////////////////////////////////////////////////////////////////////////////////

function index_init()
{
	$(window).resize(function(){
		var height = $("html").height() - $('header').height() - $('footer').height() - 35;
		$('section').css('height', height);
		$("#index .logo").css('height', height / 1.5);
		
		if($(".logo").height() > $(".logo").width())
		{
			$(".logo > img").css('width', '100%');
			$(".logo > img").css('height', $(".logo > img").width());
		}
		else
		{
			$(".logo > img").css('height', '100%');
			$(".logo > img").css('width', $(".logo > img").height());
		}
	});
	$(window).trigger('resize');
}

function select_init()
{
	footer_none();
	click_back();
	click_title_dropbox();
	click_pagemove("select_detail.html?type1=");
}

function select_detail_init()
{
	footer_none();
	click_back();
	click_title_dropbox();
	var para = location.href.split("?")[1];
	click_pagemove("image.html?" + para + "&type2=");
}

function image_init()
{
	footer_none();
	click_back();
	click_title_dropbox();
	dragsetting();
	var para = location.href.split("?")[1];
	click_submit("", para);
}

function analyze_init()
{
	nav_none_footer_have();
	click_back();
	click_title_dropbox();
	analyze_contentsHeight();
	var para = location.href.split("?")[1];
	click_pagemove('result.html?' + para + "&disease=");
	click_cropmove();
}

function result_init()
{
	more_have_footer_have();
	click_back();
	click_title_dropbox();
	result_hidden();
	click_result_append();
	click_cropmove();
	click_prescription();
}

function crop_image_init()
{
	footer_none();
	click_back();
	click_title_dropbox();
	jcrop_setting();
	//var para = location.href.split("?")[1];
	//click_submit("", para);
}

function crop_analyze_init()
{
	footer_none();
	click_back();
	click_title_dropbox();
	crop_analyze_contentsHeight();
	var para = location.href.split("?")[1];
	click_pagemove("crop_result.html?" + para + "&disease=");
}

function crop_result_init()
{
	more_have_footer_none();
	click_back();
	click_title_dropbox();
	result_hidden();
	click_result_append();
	click_prescription();
}
function prescription_init()
{
	nav_none_footer_none();
	click_title_dropbox();
	click_prescription();
}
//////////////////////////////////////////////////////////////////////////////

function footer_have()
{
	$(window).resize(function(){
		var height = $("html").height() - $('header').height() - $('nav').height() - $('footer').height() - 53;
		$('section').css('height', height);
	});
	$(window).trigger('resize');
}

function footer_none()
{
	$(window).resize(function(){
		var height = $("html").height() - $('header').height() - $('nav').height() - 35;
		$('section').css('height', height);
	});
	$(window).trigger('resize');
}
function nav_none_footer_none()
{
	$(window).resize(function(){
		var height = $("html").height() - $('header').height();
		$('section').css('height', height);
	});
	$(window).trigger('resize');
}
function nav_none_footer_have()
{
	$(window).resize(function(){
		var height = $("html").height() - $('header').height() - $('footer').height() - 28;
		$('section').css('height', height);
	});
	$(window).trigger('resize');
}

function more_have_footer_have()
{
	$(window).resize(function(){
		var height = $("html").height() - $('header').height() - $('#more-box').height() - $('footer').height() - 28;
		$('section').css('height', height);
	});
	$(window).trigger('resize');
}
function more_have_footer_none()
{
	$(window).resize(function(){
		var height = $("html").height() - $('header').height() - $('#more-box').height()-30;
		$('section').css('height', height);
	});
	$(window).trigger('resize');
}
function analyze_contentsHeight()
{
	$(window).resize(function(){
		var h = $('#uploaded').height()-6;
		$('.output-sim').css("height", h + "px");
		$('.output-video').css("height", h + "px");
	});
	$(window).trigger('resize');
}
function crop_analyze_contentsHeight()
{
	$(window).resize(function(){
		var h = $('#uploaded').height()-15;
		$('.output-sim').css("height", h + "px");
		$('.output-video').css("height", h + "px");
	});
	$(window).trigger('resize');
}
function result_hidden()
{
	var rowCnt = Math.floor($("section").width()/280) - 1;
	console.log("asd");
	console.log(rowCnt);
	
	var list = $("section").find("article");
	var init_max = 0;

	if(rowCnt <= 3)
	{
		init_max = 6;
	}
	else
	{
		init_max = rowCnt * 2;
	}
	list.hide();

	for(var i = 0; i < init_max; i++)
	{
		list.eq(i).show();
	}
	viewCnt = init_max;

}

//////////////////////////////////////////////////////////////////////////////

function click_title_dropbox()
{
	$(".title-bar .drop-btn").click(function(){
		$(".dropbox").toggle(400);
	});
}

function click_back()
{
	$(".title-bar .back").click(function(){
		window.history.back();
	});
}

function click_pagemove(url)
{
	$(".click").click(function() {
		window.location.href = url+$(this).attr('alt');
	});
}

function click_submit(type, para)
{
	$("#submit").click(function(){
		var data = $("#viewer img")[0];
		var xhr = new XMLHttpRequest();
		xhr.upload.addEventListener("progress", function(event){ 
			
		}, false);

		xhr.onreadystatechange = function() {
			if (xhr.readyState == 4 && xhr.status == 200)
			{
				window.location.href = type + "analyze.html?" + para + "&filePath=" + xhr.responseText;
			}
		};

		var formData = new FormData(); 
		formData.append("file", data.src);
		formData.append("name", data.file.name);
		formData.append("ratio", data.height / data.width);		
		xhr.open("POST", type + "upload.html");
		xhr.send(formData);
		wrapWindowByMask();
	});
}

function click_result_append()
{
	$(".click").click(function(){
		var rowCnt = Math.floor($("section").width() / 280) - 1; // 280 : 하나의 이미지 크기, 한 row에 
		var list = $("section").find("article");
		var init_max = 0;

		if(rowCnt <= 2) // rowCnt : 한 행에 보여지는 이미지 수 (2개 이하일 경우)
		{
			init_max = 4; // 이미지를 최소 4개씩 나오게 함
		}
		else
		{
			init_max = rowCnt * 2; // 2행씩 나오게
		}

		for(var i = viewCnt; i < viewCnt + init_max; i++) // viewCnt : 현재 보여지는 이미지 개수 
		{
			try
			{
				list.eq(i).show();	
			}
			catch(e)
			{
			}
		}
		viewCnt += init_max;
		
		$("section").animate({
			scrollTop:$("section")[0].scrollHeight
		},500);
	});
}

function click_cropmove()
{
	$(".move-crop").click(function(){
		window.location.href = "crop_image.html?type1=" + $(this).attr('type1') + "&type2=" + $(this).attr('type2') + "&filePath=" + $(this).attr('img');
	});
}
function click_prescription()
{
	$(".move-prescription").click(function(){
		var move = window.location.href.split("?")[1];
		var para = window.location.href.split("?")[1].split("&");
		for(var i=0;i<para.length;i++){
			if(para[i].indexOf("filePath") != -1){
				var str = para[i].split("=")[1]
				move = move.replace(str,$("#orgimg").val());
			}				
		}
		window.location.href = "prescription.html?"+move+"&sfilePath="+$(this).attr('select');
	});
}
///////////////////////////////////////////////////////////////////////////////

function jcrop_setting()
{
	//$('#cropbox').removeAttr('style');
	var w = $('#cropbox').width();
	var h = $('#cropbox').height();
	var pos = w;

	if(w > h)
	{
		pos = h;
	}

	$('#cropbox').Jcrop({
		//bgColor: 'black',
		bgOpacity : .6,
		setSelect : [w/3, pos/4, pos/4*3, pos/4*3],
		aspectRatio : 1,
		allowResize :true,
		onSelect: function(selection)
		{
			var form = document.crop;
			form.x.value = selection.x;
			form.y.value = selection.y;
			form.w.value = selection.w;
			form.h.value = selection.h;
			form.method = "post";

			var para = location.href.split("?")[1];
			form.action = "crop_upload.html?" + para;
			form[0].type = "submit";
		}
	});
}

function dragsetting()
{
	$("#viewer").on('dragenter', function(e){
		e.stopPropagation();
		e.preventDefault();
		$(this).css('border', '2px solid #5272A0');
	});

	$("#viewer").on('dragleave', function(e){
		e.stopPropagation();
		e.preventDefault();
		$(this).css('border', '3px dashed #A4A4A4');
	});

	$("#viewer").on('dragover', function(e){
		e.stopPropagation();
		e.preventDefault();
	});

	$("#viewer").on('drop', function(e){
		e.preventDefault();
		$(this).css('border', '2px dotted #8296C2');
		$("#viewer").empty();
		var files = e.originalEvent.dataTransfer.files;
		preview(files);
	});

	$("#viewer").click(function(){
		$("#file").click();
	});

	$("#file").change(function(){
		$("#viewer").empty();
		preview(this.files);
	});
}

///////////////////////////////////////////////////////////////////////////////

function preview(files)
{
	if(files == null)
	{
		return;
	}
	if(files.length < 1)
	{
		return;
	}

	var file = files[0];
	
	if(file.type.startsWith("image/"))
	{
		var img = document.createElement("img");
		img.file = file;
		img.style.width = "100%";

		$("#viewer").append(img);
		$("#viewer").css("border-style","none");

		var reader = new FileReader();
		
		reader.onload = (function(aImg){
			return function(e){
				aImg.src = e.target.result;
			};
		})(img);
		reader.readAsDataURL(file);

		$("#submit").css("display","block");
	}
	else
	{
		alert("이미지를 넣어주세요");
	}
}

function wrapWindowByMask()
{
	$('.mask').css("display", "block");
	$('.load-box').css("display", "block");

	$(window).resize(function(){
		var h = $(window).height() / 2;
		var w = $(window).width() / 2;

		$('.load-box').css("top", h - $('.load-box').height() / 2);
		$('.load-box').css("left", w - $('.load-box').width() / 2);
	});
	$(window).trigger('resize');
}

///////////////////////////////////////////////////////////////////////////////
