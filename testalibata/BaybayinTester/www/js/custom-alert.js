function customAlert(titleAlert, messageAlert) {
	try {
		navigator.notification.alert(messageAlert, null, titleAlert, 'Ok');
	} catch (err) {
		alert(messageAlert);
	}
}