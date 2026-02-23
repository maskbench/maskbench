1- Ensure Github is connected and you can import images.
* Create [Github Token](https://github.com/settings/tokens). Ensure read:packages is allowed.
* To add credentials:
    ```bash
    mkdir -p ~/.config/enroot
    nano ~/.config/enroot/.credentials
    ```
    Enter the following then save the file:
    ```bash
    machine ghcr.io login your_github_username password ghp_your_token_here
    ```
2- Run ```nano .env``` and Uncomment Enroot specific variables.

3- ```cd enroot_scripts```

**Note:** you may have to run ```chmod +x file_name``` if permission denied.

4- Run the following:
```bash
./import_images.sh
./start_services.sh
```
5- You can run ```./check_services.sh``` to check status of images.

6- Run ```./start_runner_service.sh``` to run maskbench. You may need to run ```stty echo``` if commands are not visible.

7- Run ```./stop_services.sh``` once finished to stop the services.