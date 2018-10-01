var curent_character = '';
var at_index = 0;
var full_name = '';
var all_chars = [
	'a', 'e_i', 'o_u',
	'ba', 'be_bi', 'bo_bu', 'b',
	'ka', 'ke_ki', 'ko_ku', 'k',
	'da_ra', 'de_di', 'do_du', 'd',
	'ga', 'ge_gi', 'go_gu', 'g',
	'ha', 'he_hi', 'ho_hu', 'h',
	'la', 'le_li', 'lo_lu', 'l',
	'ma', 'me_mi', 'mo_mu', 'm',
	'na', 'ne_ni', 'no_nu', 'n',
	'nga', 'nge_ngi', 'ngo_ngu', 'ng',
	'pa', 'pe_pi', 'po_pu', 'p',
	'sa', 'se_si', 'so_su', 's',
	'ta', 'te_ti', 'to_tu', 't',
	'wa', 'we_wi', 'wo_wu', 'w',
	'ya', 'ye_yi', 'yo_yu', 'y',
];

var canvas, context;

$(document).ready(function() {
	var $window = $(window);
	var temp_width = $window.width();
	var temp_height = $window.height();
	var temp_dim = (temp_width<temp_height) ? temp_width : temp_height;
	temp_dim = temp_dim - 50;
	$("#sheet").attr("width", temp_dim);
	$("#sheet").attr("height", temp_dim);
	context = document.getElementById('sheet').getContext("2d");
	canvas = document.getElementById('sheet');
	context = canvas.getContext("2d");
	context.strokeStyle = "#000000";
	context.lineJoin = "round";
	context.lineWidth = temp_dim/14;
	context.fillStyle = "white";
	context.fillRect(0, 0, canvas.width, canvas.height);
	canvas.addEventListener('mousedown', mouseWins);
	canvas.addEventListener('touchstart', touchWins);
	if (temp_dim<=480) {
		$('.w3-button').removeClass("w3-xlarge");
	}
	$('#loader-container').hide();
});

function submitAndUpload() {
	$('#loader-container').show();
	var res = [];
	//grab the context from your destination canvas
	var dest_context = document.getElementById('sheet-small').getContext("2d");
	var source_canvas = document.getElementById('sheet');
	dest_context.drawImage(source_canvas, 0, 0,28,28);
	// Generate the image data
    var pic = document.getElementById("sheet-small").toDataURL("image/png");
    pic = pic.replace(/^data:image\/(png|jpg);base64,/, "");
    // Sending the image data to Server
    $.post("http://127.0.0.1:8080/classify-image", {'curent_character':curent_character, 'imageData':pic}, function(result) {
		for (var x=0; x<result.all_chars.length; x++) {
			res.push([result.all_chars[x], result.result_float[x]]);
		}
		res = res.sort(function(a,b) {
		    return b[1] - a[1];
		});
		$('#five-results').html('');
		for (var x=0; x<5; x++) {
			$('#five-results').append('<tr>');
			$('#five-results').append('<td>'+res[x][0]+'</td>');
			$('#five-results').append('<td>'+(res[x][1]*100).toFixed(10)+'%</td>');
			$('#five-results').append('</tr>');
		}
		$('#loader-container').hide();
	});
	clearCanvas();
}