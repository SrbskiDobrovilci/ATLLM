from gigachat import GigaChat

giga = GigaChat(
   credentials="MDE5YWVkMzItOGJjNy03ODY2LWExZjItZTZmZmJkYmU1NTU3OmViNDM0ZWEwLTE0ODgtNGVhZC05YjA5LWNmOTZkMDgwYmQ5ZA==",
   scope="GIGACHAT_API_PERS",
   model="GigaChat",
   ca_bundle_file="russian_trusted_root_ca_pem.crt"
)

response = giga.get_token()

print(response.access_token)