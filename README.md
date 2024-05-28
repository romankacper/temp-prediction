# Aplikacja internetowa do przewidywania temperatury powietrza dla Delhi

## Opis
Aplikacja pythona tworzy model uczenia maszynowego, i przedstawia wyniki jego działania na wykresie, oparty na warstwach LSTM, który przewiduje temperaturę powietrza (w stopniach Celsjusza) w mieście Delhi (Indie) na przyszły dzień na podstawie danych pogodowych – temperatura powietrza, wilgotność powietrza, prędkość wiatru, ciśnienie atmosferyczne – z 14 ostatnich dni.

## Instalacja
Wymagane zainstalowane lokalnie środowisko Dockera i narzędzie Docker Compose.

## Jak używać
W wierszu poleceń przejść do katalogu projektu i wywołać polecenie: `docker-compose up`. W przypadku komputera z systemem macOS należy wywołać inne polecenie: `docker-compose -f docker-compose-arm.yml up`.
System zostanie uruchomiony przez Dockera i zostanie wskazany adres, pod którym działa aplikacja internetowa.
